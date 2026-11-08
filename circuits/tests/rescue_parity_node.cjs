// Trained stimulus-reset export: Node decisions/counts on all 512 rescue validation trials,
// and exact event/delivery traces on the first 24, must equal the float64 Python reference.
const fs=require('node:fs'),assert=require('node:assert/strict'),path=require('node:path');
const core=require('../circuit_core.js'),root=path.resolve(__dirname,'..');
const model=JSON.parse(fs.readFileSync(path.join(root,'results/rescue/order_random_reset_nopen_u600_stage3_final_hard.json'),'utf8'));
const cases=JSON.parse(fs.readFileSync(path.join(root,'results/rescue/parity_reset_trained.json'),'utf8'));
assert.equal(model.clock.mode,'stimulus_reset');
let traced=0;
cases.forEach((c,i)=>{
  const js=core.infer(model,c.example,{trace:!!c.events});
  for(const k of ['counts','prediction','exit_events','clock_origin'])assert.deepStrictEqual(js[k],c[k],`case ${i} ${k}`);
  if(c.events){
    assert.deepStrictEqual(js.events.map(e=>[e.yard,e.exit_tick,e.release_tick,e.phase]),c.events,`case ${i} events`);
    assert.deepStrictEqual(js.deliveries,c.deliveries,`case ${i} deliveries`);traced++;
  }
});
console.log(`RESET PARITY PASS: ${cases.length} trained-model decisions, ${traced} exact traces`);

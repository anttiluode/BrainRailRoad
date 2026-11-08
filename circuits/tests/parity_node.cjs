const fs = require('node:fs'), assert=require('node:assert/strict'),path=require('node:path');
const core=require('../circuit_core.js'),root=path.resolve(__dirname,'..');
const model=JSON.parse(fs.readFileSync(path.join(root,'results/untrained_seed11.json'),'utf8'));
const examples=JSON.parse(fs.readFileSync(path.join(root,'results/parity_cases.json'),'utf8'));
let nEvents=0;
for(let i=0;i<examples.length;i++){
  const f=examples[i],py=f.result,js=core.infer(model,f.example,{...f.options,trace:true});
  for(const key of ['counts','prediction','label','correct','abstained','active_rails','exit_events','internal_deliveries','beyond_horizon_deliveries']){
    assert.deepStrictEqual(js[key],py[key],`Case ${i} ${key}`);
  }
  assert.deepStrictEqual(js.deliveries,py.deliveries,`Case ${i} exact packet path/timing`);
  assert.deepStrictEqual(js.release_events,py.release_events,`Case ${i} output release timing`);
  assert.equal(js.events.length,py.events.length,`Case ${i} count`);
  for(let k=0;k<js.events.length;k++){
    for(const key of ['yard','exit_tick','release_tick','phase'])assert.equal(js.events[k][key],py.events[k][key],`Case ${i} event ${k} ${key}`);
    for(const key of ['radius','load','gain'])assert.ok(Math.abs(js.events[k][key]-py.events[k][key])<1e-10,`Case ${i} ${k} ${key}`);
    for(let j=0;j<2;j++)assert.ok(Math.abs(js.events[k].vector[j]-py.events[k].vector[j])<1e-10,`Case ${i} ${k} z`);
  }
  nEvents+=js.events.length;
}
console.log(`PARITY PASS: ${examples.length} fixtures; ${nEvents} internal events; exact routing/releases, 1e-10 float tolerance.`);

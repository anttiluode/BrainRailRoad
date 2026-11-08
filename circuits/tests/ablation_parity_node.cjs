const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const rail=require('../circuit_core.js'),R=path.resolve(__dirname,'..');
const load=f=>JSON.parse(fs.readFileSync(path.join(R,'results',f),'utf8'));
const mods={fixed_phase:load('fixed_phase_reference.json'),spiking:load('spiking_reference.json')};
for(const c of load('ablation_parity_cases.json')){
  const r=rail.infer(mods[c.model],c.example,{trace:true});
  for(const [k,expect] of [['counts',c.counts],['prediction',c.prediction],['exit_events',c.exit_events],['internal_deliveries',c.deliveries],['first_releases',c.first_releases]]){
    assert.deepStrictEqual(k==='first_releases'?r.release_events.slice(0,8):r[k],expect,`${c.model} ${c.example.order}/${c.example.context} ${k}`);
  }
}
console.log('ABLATION PARITY PASS: fixed phase + signed time-blind spiking, 8 deterministic cases');

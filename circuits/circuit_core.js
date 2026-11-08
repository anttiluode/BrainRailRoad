/* BrainRailRoad hard inference core. Dependency free in Node or file:// browser.
 * Exactly one unit event per yard per 32-tick exit. No learned gradient proxies.
 * Canonical IDs A=0, B=1, H1=2..7, H2=8..13, O0/O1=14/15.
 */
(function (root, factory) {
  const x = factory();
  if (typeof module === 'object' && module.exports) module.exports = x;
  else root.BrainRail = x;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const PI = Math.PI;
  const ANGLE = [0, PI / 2, PI, 3 * PI / 2];
  const FAMILIES = ['source_h1', 'h1_h2', 'h2_h2', 'h2_output'];
  const QUOTAS = {source_h1: 1, h1_h2: 3, h2_h2: 1, h2_output: 3};
  const DELAYS = [1, 3, 5, 7, 9, 11, 13, 15];
  const TAU = [0, 8, 16, 24];
  function candidates() {
    const es = [];
    function add(family, dst, sources) {
      sources.forEach((src, i) => es.push({id: es.length, family, src, dst, port: i % 4}));
    }
    for (let d = 2; d < 8; d++) add('source_h1', d, [0, 1]);
    for (let d = 8; d < 14; d++) add('h1_h2', d, [2, 3, 4, 5, 6, 7]);
    for (let d = 8; d < 14; d++) add('h2_h2', d, [8, 9, 10, 11, 12, 13].filter(s => s !== d));
    for (let d = 14; d < 16; d++) add('h2_output', d, [8, 9, 10, 11, 12, 13]);
    return es;
  }
  function validate(model) {
    if (model.schema_version !== 1 || !Array.isArray(model.edges) || model.edges.length !== 90)
      throw Error('Invalid export: require schema 1 and 90 candidate rails');
    const ref = candidates();
    let active = 0;
    for (let i = 0; i < 90; i++) {
      const e = model.edges[i], a = ref[i];
      for (const k of ['id', 'src', 'dst', 'family', 'port']) if (e[k] !== a[k]) throw Error('Noncanonical edge '+i+' '+k);
      if (!DELAYS.includes(e.delay) || !(model.architecture?.spiking ? e.weight >= -2 && e.weight <= 2 : e.weight >= 0.05 && e.weight <= 2)) throw Error('Invalid operational rail '+i);
      if (e.active && e.broken) throw Error('Active broken rail '+i);
      active += Number(!!e.active);
    }
    if (active !== 36) throw Error('Expected 36 active rails, got '+active);
    for (const fam of FAMILIES) {
      const destinations = [...new Set(model.edges.filter(e => e.family === fam).map(e => e.dst))];
      for (const dst of destinations)
        if (model.edges.filter(e => e.family === fam && e.dst === dst && e.active).length !== QUOTAS[fam])
          throw Error('Incoming quota violated '+fam+':'+dst);
    }
    if (model.thresholds.length !== 14 || model.context.length !== 12) throw Error('Bad yard parameters');
    const mode = model.clock?.mode ?? 'absolute';
    if (!['absolute','stimulus_reset'].includes(mode)) throw Error('Unknown clock mode '+mode);
    if (mode !== 'absolute' && model.architecture?.spiking) throw Error('Spiking comparator has no exit clock to reset');
    return true;
  }
  function trial(order, context, onset, gap, noiseA=0, noiseB=0) {
    if (![0,1].includes(order) || ![0,1].includes(context)) throw Error('Order and context are bits');
    return {order,context,onset,gap,noise_a:noiseA,noise_b:noiseB,
      a_tick:onset+(order===0?0:gap)+noiseA,
      b_tick:onset+(order===0?gap:0)+noiseB, label:order^context};
  }
  function infer(model, example, opts={}) {
    if (opts.validate !== false) validate(model);
    if(model.architecture?.spiking) return inferSpiking(model, example, opts);
    const ticks=opts.ticks??model.ticks;
    const broken = new Set(opts.brokenEdges||[]);
    const bySource=Array.from({length:16},()=>[]);
    const es=model.edges;
    let live=0;
    for(let i=0;i<es.length;i++) {
      if(!es[i].active||es[i].broken||broken.has(i))continue;
      bySource[es[i].src].push(i);live++;
    }
    const arrivals=new Map(), releases=new Map();
    const events=[],deliveries=[],outputReleases=[];
    let beyond=0;
    function schedule(map,t,payload){if(!map.has(t))map.set(t,[]);map.get(t).push(payload);}
    for(const [src,sourceTick] of [[0,example.a_tick],[1,example.b_tick]]){
      for(const eid of bySource[src]){
        const e=es[eid],at=sourceTick+e.delay;
        schedule(arrivals,at,[eid,e.weight]);
        if(at>=ticks)beyond++;
        if(opts.trace)deliveries.push({edge:eid,source:src,dest:e.dst,tick:at,origin:'sensory'});
      }
    }
    const L=new Float64Array(16),zx=new Float64Array(16),zy=new Float64Array(16),b=new Float64Array(16),c=new Float64Array(16);
    const baskets=opts.basket||{},chandeliers=opts.chandelier||{};
    // Optional stimulus-locked reset: explicit pilot at the first source
    // emission sets every yard's phase origin (see model.py clock_origin).
    const resetClock=(model.clock?.mode ?? 'absolute')==='stimulus_reset';
    const origin=resetClock?Math.min(example.a_tick,example.b_tick):0;
    const mod=(x,n)=>((x%n)+n)%n;
    for(let t=0;t<ticks;t++) {
      const vals=new Map(), incoming=new Map();
      for(const [eid,value] of arrivals.get(t)||[])vals.set(eid,(vals.get(eid)||0)+value);
      arrivals.delete(t);
      for(const [eid,value] of vals.entries()){
        const dst=es[eid].dst;
        if(!incoming.has(dst))incoming.set(dst,[]);
        incoming.get(dst).push([eid,value]);
      }
      const theta=2*PI*mod(t-origin,model.period)/model.period;
      const ct=Math.cos(theta),st=Math.sin(theta);
      const exitTick=(t>origin && mod(t-origin,model.period)===0);
      for(let y=2;y<16;y++) {
        b[y]=model.basket_leak*b[y]+Number(baskets[y]?.[t]||0);
        c[y]=model.chandelier_leak*c[y]+Number(chandeliers[y]?.[t]||0);
        let basal=0;
        for(const [eid,v] of incoming.get(y)||[]){
          const gate=Math.exp(model.kappa*(Math.cos(theta-ANGLE[es[eid].port])-1));
          basal+=gate*v;
        }
        const shunt=Math.exp(-model.basket_strength*b[y]);
        const lstar=(model.leak*L[y]+basal)*shunt;
        const xstar=(model.leak*zx[y]+basal*ct)*shunt;
        const ystar=(model.leak*zy[y]+basal*st)*shunt;
        L[y]=lstar;zx[y]=xstar;zy[y]=ystar;
        if(!exitTick)continue;
        const gain=y<14?Math.exp(model.context_scale*Math.tanh(model.context[y-2][example.context])):1;
        const radius=Math.sqrt(xstar*xstar+ystar*ystar+model.epsilon*model.epsilon);
        const threshold=model.thresholds[y-2];
        const fire=(gain*radius>=threshold && lstar>model.epsilon && c[y]<model.veto_threshold);
        if(!fire)continue;
        const scores=[xstar/radius,ystar/radius,-xstar/radius,-ystar/radius];
        let address=0;
        for(let k=1;k<4;k++)if(scores[k]>scores[address])address=k; // lowest index wins ties
        if(model.architecture?.fixed_phase) address=0;
        if(opts.phaseOverride!==undefined && opts.phaseOverride!==null){
          address= typeof opts.phaseOverride==='number' ? opts.phaseOverride%4 : (opts.phaseOverride[address]??address)%4;
        }
        const release=t+model.address_delays[address];
        const keep=Math.max(0,1-threshold/(gain*radius));
        L[y]*=keep;zx[y]*=keep;zy[y]*=keep;
        events.push({yard:y,exit_tick:t,release_tick:release,phase:address,
                     vector:[xstar,ystar],radius,load:lstar,gain});
        schedule(releases,release,[y,address,t]);
        for(const eid of bySource[y]){
          const e=es[eid],at=release+e.delay;
          schedule(arrivals,at,[eid,e.weight]);
          if(at>=ticks)beyond++;
          if(opts.trace)deliveries.push({edge:eid,source:y,dest:e.dst,tick:at,origin:'yard'});
        }
      }
      for(const [y,k,exitAt] of releases.get(t)||[]){
        if(y===14||y===15)outputReleases.push({yard:y,tick:t,phase:k,exit_tick:exitAt});
      }
      releases.delete(t);
    }
    const [start,stop]=model.score_window;
    const counts=[14,15].map(y=>outputReleases.filter(r=>r.yard===y&&r.tick>=start&&r.tick<stop).length);
    const prediction=counts[0]===counts[1]?null:(counts[1]>counts[0]?1:0);
    const result={counts,prediction,label:example.label,correct:prediction===example.label,
      abstained:prediction===null,active_rails:live,exit_events:events.length,
      internal_deliveries:opts.trace?deliveries.length:null,beyond_horizon_deliveries:beyond,
      release_events:outputReleases};
    if(resetClock){result.clock_origin=origin;result.pilot_deliveries=14;}
    if(opts.trace){result.events=events;result.deliveries=deliveries;}
    return result;
  }
  function inferSpiking(model,example,opts={}) {
    // Same classical signed LIF-style comparator as Python infer_spiking.
    const ticks=opts.ticks??model.ticks,broken=new Set(opts.brokenEdges||[]),es=model.edges;
    const bySource=Array.from({length:16},()=>[]);let active=0;
    for(let i=0;i<es.length;i++){
      if(!es[i].active||es[i].broken||broken.has(i))continue;
      bySource[es[i].src].push(i);active++;
    }
    const arrivals=new Map(),events=[],deliveries=[],outputReleases=[];
    let beyond=0;
    function schedule(t,payload){if(!arrivals.has(t))arrivals.set(t,[]);arrivals.get(t).push(payload);}
    for(const [src,t] of [[0,example.a_tick],[1,example.b_tick]])for(const eid of bySource[src]){
      const e=es[eid],at=t+e.delay;schedule(at,[eid,e.weight]);
      if(at>=ticks)beyond++;
      if(opts.trace)deliveries.push({edge:eid,source:src,dest:e.dst,tick:at,origin:'sensory'});
    }
    const v=new Float64Array(16),lastFire=new Int32Array(16).fill(-10000);
    for(let t=0;t<ticks;t++){
      const byDest=new Map();
      for(const [eid,value] of arrivals.get(t)||[]){const d=es[eid].dst;
        if(!byDest.has(d))byDest.set(d,[]);
        byDest.get(d).push(value);
      }
      arrivals.delete(t);
      for(let y=2;y<16;y++){
        let basal=0;
        for(const value of byDest.get(y)||[])basal+=value;
        const next=model.leak*v[y]+basal;
        const gain=y<14?Math.exp(model.context_scale*Math.tanh(model.context[y-2][example.context])):1;
        const h=model.thresholds[y-2];
        const fire=(gain*next>=h && t-lastFire[y]>=2);
        if(!fire){v[y]=next;continue;}
        v[y]=next-h/gain;lastFire[y]=t;
        events.push({yard:y,exit_tick:t,release_tick:t,phase:0,vector:[0,0],radius:next,load:next,gain});
        if(y===14||y===15)outputReleases.push({yard:y,tick:t,phase:0,exit_tick:t});
        for(const eid of bySource[y]){
          const e=es[eid],at=t+e.delay;
          schedule(at,[eid,e.weight]);
          if(at>=ticks)beyond++;
          if(opts.trace)deliveries.push({edge:eid,source:y,dest:e.dst,tick:at,origin:'yard'});
        }
      }
    }
    const [start,stop]=model.score_window;
    const counts=[14,15].map(y=>outputReleases.filter(r=>r.yard===y&&r.tick>=start&&r.tick<stop).length);
    const prediction=counts[0]===counts[1]?null:(counts[1]>counts[0]?1:0);
    const result={counts,prediction,label:example.label,correct:prediction===example.label,
      abstained:prediction===null,active_rails:active,exit_events:events.length,
      internal_deliveries:opts.trace?deliveries.length:null,beyond_horizon_deliveries:beyond,
      release_events:outputReleases};
    if(opts.trace){result.events=events;result.deliveries=deliveries;}
    return result;
  }
  return {candidates,validate,trial,infer,TAU,DELAYS,QUOTAS};
});

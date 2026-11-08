"""CPU PyTorch surrogate-BPTT for the EXACT hard 320-tick forward.

Every exit evaluates the full 90-candidate topology's surrogate output,
including silent yards/dormant rails, and schedules all 4x8 phase/delay slots.
No differentiable teacher state appears as a hidden payload. Gradient is an ST
estimator, **not** the derivative of this discontinuous network or a local rule.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from model import ROOT,CFG,YARDS,OUTPUTS,DELAYS,TAU,candidate_edges,make_model,apply_topology

# Tiny workloads: extra threading can dominate a B=4 or B=32 simulation.
torch.set_default_dtype(torch.float64)

def logit(z:float)->float:
    return math.log(z/(1-z))

class SurrogateRail(nn.Module):
    def __init__(self,seed:int=11,fixed_phase=False,fixed_delay=False,fixed_topology=False,spiking=False,init_threshold=0.35,
                 clock_reset=False,surrogate_through_empty=False):
        super().__init__()
        if clock_reset and spiking:raise ValueError('the spiking comparator has no exit clock to reset')
        self.model_seed=seed
        # Optional stimulus-locked pilot reset (model.py clock_origin). Default off = original semantics.
        self.clock_reset=clock_reset
        # DIAGNOSTIC ONLY (rescue.py gradient probe): drop the multiplicative
        # empty-yard mask from the surrogate pulse. Hard forward is unchanged
        # because hard_fire already requires L>eps. Never used for training.
        self.surrogate_through_empty=surrogate_through_empty
        self.fixed_phase=fixed_phase;self.fixed_delay=fixed_delay;self.fixed_topology=fixed_topology;self.spiking=spiking
        reference=make_model(seed)
        es=reference['edges']
        self.register_buffer('src',torch.tensor([x['src'] for x in es],dtype=torch.long))
        self.register_buffer('dst',torch.tensor([x['dst']-2 for x in es],dtype=torch.long))
        self.register_buffer('port',torch.tensor([x['port'] for x in es],dtype=torch.long))
        self.register_buffer('mask_start',torch.tensor([float(x['active']) for x in es]))
        self.register_buffer('intact',torch.ones(90))
        self.register_buffer('source_yard',self.src[12:]-2)
        self.register_buffer('dst_mat',F.one_hot(self.dst,14).to(torch.float64))
        self.register_buffer('sin_ports',torch.tensor([0.0,1.0,0.0,-1.0])[self.port])
        self.register_buffer('cos_ports',torch.tensor([1.0,0.0,-1.0,0.0])[self.port])
        self.register_buffer('delay_values',torch.tensor(DELAYS,dtype=torch.long))
        self.register_buffer('hidden_context_yards',torch.tensor(list(range(12)),dtype=torch.long))
        self.groups=[]
        for family in ('source_h1','h1_h2','h2_h2','h2_output'):
            for dst in sorted({e['dst'] for e in es if e['family']==family}):
                indices=[i for i,e in enumerate(es) if e['family']==family and e['dst']==dst]
                self.groups.append((indices,{'source_h1':1,'h1_h2':3,'h2_h2':1,'h2_output':3}[family]))
        # Bounded operational weights and thresholds; no clipping after optimizer.
        wg=[logit((e['weight']-0.05)/1.95) for e in es]
        self.raw_weight=nn.Parameter(torch.tensor(wg))
        self.raw_threshold=nn.Parameter(torch.full((14,),logit((init_threshold-0.15)/1.35)))
        self.context=nn.Parameter(torch.tensor(reference['context']))
        self.score=nn.Parameter(torch.tensor([e['score'] for e in es]))
        gen=torch.Generator().manual_seed(seed*271+19)
        self.delay_logits=nn.Parameter(torch.randn(90,8,generator=gen)*0.1)
        if fixed_delay:self.delay_logits.requires_grad_(False)
        if fixed_topology:self.score.requires_grad_(False)
        if spiking:
            # Signed weights for the spiking comparator are allowed in [-2,2].
            self.raw_weight=nn.Parameter(torch.tensor([math.atanh(e['weight']/2) for e in es]))
        self.stats={}

    def hard_mask(self):
        out=torch.zeros(90,dtype=self.score.dtype,device=self.score.device)
        for indices,k in self.groups:
            eligible=[i for i in indices if self.intact[i].item()>0.5]
            if len(eligible)<k:raise ValueError('Broken-edge quota unsatisfiable')
            picks=sorted(eligible,key=lambda i:(-self.score[i].item(),i))[:k]
            out[picks]=1.0
        return out

    def surrogate_mask(self):
        hard=self.hard_mask()
        if self.fixed_topology:return self.mask_start*self.intact
        soft=torch.sigmoid(self.score)
        return (hard-soft.detach()+soft)*self.intact

    def operational_weights(self):
        if self.spiking:return 2.0*torch.tanh(self.raw_weight)
        return 0.05+1.95*torch.sigmoid(self.raw_weight)

    def operational_thresholds(self):
        return 0.15+1.35*torch.sigmoid(self.raw_threshold)

    def _delay_st(self,temp):
        soft=F.softmax(self.delay_logits/temp,dim=-1)
        hard=F.one_hot(self.delay_logits.argmax(dim=-1),8).to(soft.dtype)
        return hard-soft.detach()+soft

    def forward(self, a_tick:torch.Tensor,b_tick:torch.Tensor,context:torch.Tensor,
                temperature_fire=0.25,temperature_phase=0.5,temperature_delay=1.0,
                return_metrics=False):
        """Hard forward, surrogate backprop, one full independent trial per batch row.

        All 90 candidate rail operations (including dormant ones) contribute to
        surrogate learning. Queue arrivals are *already weighted*; port capture
        multiplies by the gate but does NOT apply weight again.
        """
        assert a_tick.ndim==b_tick.ndim==context.ndim==1
        B=a_tick.shape[0];device=self.raw_weight.device;T=320
        mask=self.surrogate_mask();weights=self.operational_weights();threshold=self.operational_thresholds()
        delay=self._delay_st(temperature_delay)
        # At source timestamps A/B release immediately; source->H1 delivery
        # depends on the actual selected hard delay and ST alternative slots.
        # Source indicator is fixed task data, never a trainable clock.
        source_times=torch.where(self.src[:12][None,:]==0,a_tick[:,None],b_tick[:,None])
        arrivals_source=source_times[:,:,None]+self.delay_values[None,None,:]
        source_w=(mask[:12]*weights[:12])[None,:,None]
        source_current=[None]*T
        for t in range(T):
            if t < 64 or t > 140:continue
            match=(arrivals_source==t).to(weights.dtype)
            if bool(match.any()):source_current[t]=(match*delay[None,:12,:]*source_w).sum(dim=-1)
        # Hard-event inference carries no source-side analog vector.
        queue=[[] for _ in range(T)]
        L=torch.zeros(B,14,dtype=weights.dtype,device=device)
        zx=torch.zeros_like(L);zy=torch.zeros_like(L)
        C=torch.zeros(B,2,dtype=weights.dtype,device=device)
        # Teacher-independent constant context bus into hidden yards ONLY.
        a=torch.cat([torch.exp(0.7*torch.tanh(self.context[:,context].T)),
                     torch.ones(B,2,dtype=weights.dtype,device=device)],dim=1)
        empty78=torch.zeros(B,78,dtype=weights.dtype,device=device)
        empty12=torch.zeros(B,12,dtype=weights.dtype,device=device)
        fires_sum=torch.zeros((),dtype=weights.dtype,device=device)
        copy_sum=mask[:12].sum().expand(B).sum()
        past_fires=torch.full((B,14),-10000,dtype=torch.long,device=device)
        origin=torch.minimum(a_tick,b_tick) if self.clock_reset else None
        for t in range(T):
            active_input=source_current[t] if source_current[t] is not None else empty12
            incoming_hidden=torch.stack(queue[t]).sum(dim=0) if queue[t] else empty78
            if self.clock_reset:
                # Per-trial phase origin; per-row clock and per-row exit flag.
                rel=t-origin
                ph=torch.remainder(rel,32)
                th=2*math.pi*ph.to(weights.dtype)/32
                ct=torch.cos(th)[:,None];st=torch.sin(th)[:,None]
                exit_row=((rel>0)&(ph==0)).to(weights.dtype)[:,None]
                g=torch.exp(2.0*(ct*self.cos_ports[None,:]+st*self.sin_ports[None,:]-1))
            else:
                theta=2*math.pi*(t%32)/32
                ct=math.cos(theta);st=math.sin(theta);exit_row=None
                g=torch.ones_like(self.cos_ports) if self.spiking else torch.exp(2.0*(math.cos(theta)*self.cos_ports+math.sin(theta)*self.sin_ports-1))
            basal=(active_input*g[...,:12])@self.dst_mat[:12]+(incoming_hidden*g[...,12:])@self.dst_mat[12:]
            # Main training task has no inhibitor interventions.
            if self.spiking:
                Lstar=0.985*L+basal
                L=Lstar
                zx=zx*0;zy=zy*0
            else:
                Lstar=0.985*L+basal
                xstar=0.985*zx+basal*ct
                ystar=0.985*zy+basal*st
                L=Lstar;zx=xstar;zy=ystar
            if exit_row is not None:
                if not bool(exit_row.any()):continue
            elif not self.spiking and (t==0 or t%32):continue
            if self.spiking:
                radius=Lstar
                hard_fire=((a*radius>=threshold[None,:]) & (t-past_fires>=2)).to(weights.dtype)
            else:
                radius=torch.sqrt(zx*zx+zy*zy+1e-16)
                hard_fire=((a*radius>=threshold[None,:])&(Lstar>1e-8)).to(weights.dtype)
            soft_fire=torch.sigmoid((a*radius-threshold[None,:])/temperature_fire)
            fire=hard_fire-soft_fire.detach()+soft_fire
            if self.spiking:fire=fire*(t-past_fires>=2).to(weights.dtype)
            if not self.spiking:
                if not self.surrogate_through_empty:
                    fire=fire*(Lstar>1e-8).to(weights.dtype)  # genuinely empty yards have no surrogate pulse
                if exit_row is not None:fire=fire*exit_row  # rows not at their own exit cannot fire
                q=torch.stack((zx/radius,zy/radius,-zx/radius,-zy/radius),dim=-1)
                soft_phase=F.softmax(q/temperature_phase,dim=-1)
                hard_phase=F.one_hot(q.argmax(dim=-1),4).to(weights.dtype)
                phases=hard_phase-soft_phase.detach()+soft_phase
                keep=torch.clamp(1-threshold[None,:]/(a*radius),min=0)
                L=Lstar*(1-fire*(1-keep))
                zx=zx*(1-fire*(1-keep));zy=zy*(1-fire*(1-keep))
            else:
                # Conventional tonic-current spiking comparator as defined in §7.
                # Explicit reset with *signed* weights, hard refractory 2 ticks.
                phases=torch.stack([torch.ones_like(fire),torch.zeros_like(fire),torch.zeros_like(fire),torch.zeros_like(fire)],dim=-1)
                L=Lstar-fire*threshold[None,:]/a
                past_fires=torch.where(hard_fire.bool(),torch.full_like(past_fires,t),past_fires)
            fires_sum=fires_sum+fire[:,:12].sum()
            # Count **output releases**, not AIS decisions, in [128, 320).
            for p,dt in enumerate(TAU):
                if self.fixed_phase and p>0:continue
                if 128<=t+dt<320:
                    C=C+fire[:,12:14]*phases[:,12:14,p]
            # Every exit computes all potential packets, even when no hard
            # yard fires. Each candidate edge sees all eight delay surrogates.
            payload=fire[:,self.source_yard]*weights[None,12:]*mask[None,12:]
            copy_sum=copy_sum+(fire[:,self.source_yard]*mask[None,12:]).sum()
            phase_for_edges=phases[:,self.source_yard,:]
            for p,phase_dt in enumerate(TAU):
                if self.fixed_phase and p>0:continue
                weighted=payload*phase_for_edges[:,:,p]
                for di,rail_dt in enumerate(DELAYS):
                    arrival=t+phase_dt+rail_dt
                    if arrival < T:
                        queue[arrival].append(weighted*delay[None,12:,di])
        self.stats={'hard_mask':self.hard_mask().detach(), 'hidden_events':fires_sum,
                    'deliveries':copy_sum,'hard_counts':C}
        return (C,self.stats) if return_metrics else C

    @torch.no_grad()
    def hard_export(self):
        """Export actual hard choices; nothing differentiable remains."""
        model=make_model(self.model_seed)
        m=self.hard_mask().cpu().tolist()
        w=self.operational_weights().cpu().tolist()
        delays=self.delay_logits.argmax(dim=-1).cpu().tolist()
        for i,e in enumerate(model['edges']):
            e.update(weight=float(w[i]),delay=DELAYS[delays[i]],score=float(self.score[i].item()),
                     active=bool(m[i]),broken=bool(self.intact[i].item()==0))
        model['thresholds']=[float(x) for x in self.operational_thresholds().cpu().tolist()]
        model['context']=self.context.detach().cpu().tolist()
        model['origin']='supervised-surrogate-BPTT' if not self.spiking else 'surrogate-spiking-control'
        model['architecture']={'fixed_phase':self.fixed_phase,'fixed_delay':self.fixed_delay,
                               'fixed_topology':self.fixed_topology,'spiking':self.spiking}
        if self.clock_reset:
            model['clock']={'mode':'stimulus_reset','origin':'first_source_emission',
                            'pilot':'one label-neutral broadcast to all 14 yards at the origin tick; no load, no vector',
                            'pilot_deliveries_per_trial':14}
        return model

"""Music-supervised count continuation and concrete causal layout phrases."""
import copy
from dataclasses import replace
import inspect
import torch
from torch import nn

from .expression_decoder import next_event_viability,stream_repeat_scores
from .motion_state import MOTION_WIDTH,MotionHistory,candidate_grid,motion_features
from .natural_generator import NaturalLayoutGenerator
from .factorized_model import FactorizedRiceGenerator
from .segment_labels import LABELS
from .tiny_model import TinyModelConfig


def event_context(previous, seed=None, count_only=False):
    history=previous.clone();history[:,0]=0
    if seed is not None:history=torch.cat((seed[0],history),dim=1)
    positions=torch.arange(history.shape[1],device=history.device)[None].expand_as(history)
    occupied=(history>0)&(history<16)
    last=torch.where(occupied,positions,-1).cummax(1).values
    index=last;values=[]
    for _ in range(3):
        masks=history.gather(1,index.clamp_min(0))
        masks=torch.where(index>=0,masks,0)
        if count_only:
            table=torch.tensor([m.bit_count() for m in range(16)],device=previous.device)
            values.append(nn.functional.one_hot(table[masks],5).float())
        else:values.append(nn.functional.one_hot(masks,16).float())
        before=last.gather(1,(index-1).clamp_min(0))
        index=torch.where(index>0,before,-1)
    return torch.cat(values,-1)[:,-previous.shape[1]:]


def density_supported_distribution(probability,mode,density_mean):
    """Blend with the mode when sampling would overshoot/oppose musical density."""
    values=torch.arange(1,5,device=probability.device)
    delta=(probability*values).sum(-1)-mode
    target=density_mean.clamp(1,4)-mode
    aligned=delta*target>=0
    weight=torch.where(delta.abs()>1e-6,(target.abs()/delta.abs().clamp_min(1e-6)).clamp_max(1),torch.ones_like(delta))
    weight=torch.where(aligned,weight,0.)
    anchor=nn.functional.one_hot(mode-1,4).to(probability.dtype)
    return weight[:,None]*probability+(1-weight[:,None])*anchor,weight


def selected_onset_rows(onset_probability, beats):
    """Number of music-selected rows in each represented beat, before lane edits.

    This is a density reference, not a row budget. Prefix slots participate in
    their beat; unrelated songs and beats never share a denominator.
    """
    beat_ids=torch.floor(beats+1e-7).long()
    same=beat_ids[:,:,None]==beat_ids[:,None,:]
    return (same & (onset_probability[:,None,:]>=.5)).sum(-1)


class ExpressionGenerator(NaturalLayoutGenerator):
    def __init__(self,config=TinyModelConfig(),reference_count_fallback=False,reference_layout_fallback=False,**kwargs):
        super().__init__(config,**kwargs)
        self.phrase_adapter=nn.Sequential(nn.Linear(config.width+MOTION_WIDTH+48+5,128),nn.GELU(),nn.Linear(128,16))
        nn.init.xavier_uniform_(self.phrase_adapter[0].weight);nn.init.zeros_(self.phrase_adapter[0].bias)
        nn.init.zeros_(self.phrase_adapter[-1].weight);nn.init.zeros_(self.phrase_adapter[-1].bias)
        self.expression_decoding=False
        self.stream_repetition_strength=0.
        self.count_policy='argmax'
        self.count_seed=20261112
        self.count_temperature=1.
        self._count_generator=None
        self._music_density=None
        self._selected_music_rows=None
        self._density_beats=None
        self._music_prefix=None
        self._expression_stats=None
        self.reference_counts=None
        self._reference_pass=False
        self._reference_anchor=None
        self.reference_layout_decoder=None
        if reference_count_fallback:self.enable_reference_counts(self)
        if reference_layout_fallback:self.enable_reference_layout(self)

    def enable_reference_layout(self,source):
        """Keep the original music-anchor trajectory when training layout layers."""
        self.reference_layout_decoder=copy.deepcopy(source.decoder)
        self.reference_layout_decoder.eval()
        for parameter in self.reference_layout_decoder.parameters():parameter.requires_grad_(False)

    def enable_reference_counts(self,source):
        names=('count_decoder','conditional_count_head','count_history_embedding','count_plan_projection','rhythm_history','count_style_projection')
        self.reference_counts=nn.ModuleDict({name:copy.deepcopy(getattr(source,name)) for name in names})
        self.reference_counts.eval()
        for parameter in self.reference_counts.parameters():parameter.requires_grad_(False)

    def decode_counts(self,memory,phase,condition,previous_counts,plan,count_plan_enabled):
        logits=super().decode_counts(memory,phase,condition,previous_counts,plan,count_plan_enabled)
        if self.reference_counts is None:return logits
        branch=self.reference_counts;length=previous_counts.shape[1]
        positions=torch.arange(length,device=phase.device)
        occupied=(previous_counts>0)&(previous_counts<5)
        last=torch.where(occupied,positions[None],-192).cummax(1).values
        distance=(positions[None]-last).clamp(0,192).float()/192
        features=torch.cat((nn.functional.one_hot(previous_counts.clamp_max(4),5)[...,1:].float(),distance[...,None]),-1)
        selected_plan=plan[:,:length] if count_plan_enabled else torch.zeros_like(plan[:,:length])
        hidden=branch['count_history_embedding'](previous_counts)+self.position_embedding(positions)
        hidden=hidden+self.phase_projection(phase[:,:length])+self.condition_projection(condition)[:,None]
        hidden=hidden+branch['count_plan_projection'](selected_plan)+branch['rhythm_history'](features)
        causal=torch.ones(length,length,device=phase.device,dtype=torch.bool).triu(1)
        reference=branch['conditional_count_head'](branch['count_decoder'](hidden,memory,tgt_mask=causal))
        if count_plan_enabled:reference=reference+branch['count_style_projection'](self.style_encoder(selected_plan))
        if not self.training and count_plan_enabled:
            ids=torch.where(selected_plan.sum(-1)>0,selected_plan.argmax(-1)+1,0)
            reference=reference.masked_fill(~self.style_count_allowed[ids],-torch.inf)
        if self._reference_pass:return reference
        learned=selected_plan[..., [LABELS.index('Jumpstream'),LABELS.index('Handstream')]].amax(-1)>.5
        return torch.where(learned[...,None],logits,reference)

    def decode_layout(self,memory,phase,condition,previous,plan,current_counts):
        if self._reference_pass:
            if self.reference_layout_decoder is None:
                return super().decode_layout(memory,phase,condition,previous,plan,current_counts)
            decoder=self.decoder
            try:
                self.decoder=self.reference_layout_decoder
                return super().decode_layout(memory,phase,condition,previous,plan,current_counts)
            finally:self.decoder=decoder
        _,raw=super().decode_layout(memory,phase,condition,previous,plan,current_counts)
        length=previous.shape[1]
        beats=self._motion_beats[:,:length] if self._motion_beats is not None else candidate_grid(len(previous),length,previous.device)
        motion=motion_features(previous,beats,plan[:,:length],phase[:,:length,4],self._motion_seed)
        context=torch.cat((memory[:,:length].detach(),motion,event_context(previous,self._motion_seed),
                           nn.functional.one_hot(current_counts,5).float()),-1)
        raw=raw+1.5*self.phrase_adapter(context).tanh()
        return raw.masked_fill(self.mask_counts[None,None]!=current_counts[...,None],-torch.inf),raw

    def generation_prefix_counts(self,prefix_masks):
        if self._reference_pass:return self.mask_counts[prefix_masks]
        return self._music_prefix if self._music_prefix is not None else super().generation_prefix_counts(prefix_masks)

    def generation_count_history(self,desired,placed):
        return placed if self._reference_pass else desired

    def generation_count_logits(self,memory,phase,condition,previous_counts,previous,plan,count_plan_enabled):
        logits=super().generation_count_logits(memory,phase,condition,previous_counts,previous,plan,count_plan_enabled)
        if self._reference_anchor is None:return logits
        length=previous_counts.shape[1]
        learned=plan[:,:length,[LABELS.index('Jumpstream'),LABELS.index('Handstream')]].amax(-1)>.5
        return torch.where(learned[...,None],logits,self._reference_anchor[:,:length])

    def generation_onset_logits(self,memory,condition,plan):
        logits=super().generation_onset_logits(memory,condition,plan)
        if not self._reference_pass and self.count_policy=='selected-onset-density-sample':
            self._selected_music_rows=selected_onset_rows(logits.sigmoid(),self._density_beats)
        return logits

    def refresh_density_onsets(self,probability,beats):
        if not self._reference_pass and self.count_policy=='selected-onset-density-sample':
            self._selected_music_rows=selected_onset_rows(probability,beats)

    def generation_count_choice(self,counts,conditional,memory,phase,condition,previous,generated_counts,
                                plan,position,candidate_beats,lane_state,constrained_counts):
        if self._reference_pass or self.count_policy=='argmax' or constrained_counts:return counts
        if self.count_policy not in ('music-sample','music-density-sample','selected-onset-density-sample'):raise ValueError('Unknown musical count policy.')
        selected=plan[:,position,[LABELS.index('Jumpstream'),LABELS.index('Handstream')]].amax(-1)>.5
        selected &= counts>0
        if not selected.any():return counts
        probability=conditional[selected]
        sampling=(probability.log()/self.count_temperature).softmax(-1)
        if self.count_policy in ('music-density-sample','selected-onset-density-sample'):
            if self._music_density is None:raise ValueError('Musical density sampling needs density conditions.')
            density=self._music_density[selected,position]
            rows=density[:,1]
            if self.count_policy=='selected-onset-density-sample':
                rows=self._selected_music_rows[selected,position].to(density.dtype)
                self._expression_stats['density_infeasible_rows']+=int((density[:,0]<rows).sum())
            sampling,weight=density_supported_distribution(sampling,counts[selected],density[:,0]/rows.clamp_min(.1))
            adjustment = getattr(self, 'count_density_adjustment', None)
            if adjustment is not None:
                sampling = adjustment(sampling, (density[:,0]/rows.clamp_min(.1)).clamp(1,4),
                    active=plan[selected,position,LABELS.index('Jumpstream')]>.5)
            self._expression_stats['density_guarded_rows']+=int((weight<1-1e-6).sum())
        sampled=torch.multinomial(sampling,1,generator=self._count_generator).squeeze(-1)+1
        result=counts.clone();result[selected]=sampled
        regret=probability.amax(-1).clamp_min(1e-12).log()-probability.gather(1,(sampled-1)[:,None]).squeeze(-1).clamp_min(1e-12).log()
        self._expression_stats['music_sampled_rows']+=int(selected.sum())
        self._expression_stats['music_nonmodal_rows']+=int((sampled!=counts[selected]).sum())
        self._expression_stats['sampling_log_regret_sum']+=float(regret.sum())
        return result

    def generation_layout_scores(self,logits,memory,phase,condition,previous_counts,desired,
                                 plan,position,candidate_beats,lane_state,onset_probability):
        if self._reference_pass or not self.expression_decoding or lane_state is None or not bool((desired>0).any()):return logits
        if not hasattr(lane_state,'sixteenth_runs'):return logits
        if self.stream_repetition_strength:
            adjustment=stream_repeat_scores(lane_state,candidate_beats[:,position],plan[:,position],logits)
            self._expression_stats['stream_scored_rows']+=int((adjustment<0).any(-1).sum())
            logits=logits+self.stream_repetition_strength*adjustment
        if not bool((plan[:,position,LABELS.index('Minijacks')]>.5).any()):return logits
        if len(logits)!=1:
            if self._reference_anchor is None:raise ValueError('Batched short-jack lookahead requires the protected music anchor.')
            future=onset_probability[:,position+1:]>=.5
            if not future.shape[1]:return logits
            has_future=future.any(-1);next_position=position+1+future.long().argmax(-1)
            samples=torch.arange(len(logits),device=logits.device)
            next_beat=candidate_beats[samples,next_position]
            active=(plan[:,position,LABELS.index('Minijacks')]>.5)&has_future
            active &= (next_beat-candidate_beats[:,position]<=.5+1e-7)&(plan[samples,next_position]==plan[:,position]).all(-1)
            next_count=self._reference_anchor[samples,next_position].argmax(-1)+1
            bonus=next_event_viability(lane_state,candidate_beats[:,position],next_beat,next_count,plan[:,position])
            bonus=bonus*active[:,None]
            self._expression_stats['viability_scored_rows']+=int((bonus>0).any(-1).sum())
            return logits+bonus
        current_beat=candidate_beats[:,position]
        future=(onset_probability[0,position+1:]>=.5).nonzero().flatten()
        if not len(future):return logits
        next_position=position+1+int(future[0])
        next_beat=candidate_beats[:,next_position]
        if not bool(((next_beat-current_beat)<=.5+1e-7).all()):return logits
        if not torch.equal(plan[:,next_position],plan[:,position]):return logits
        future_history=torch.cat((previous_counts,desired[:,None],torch.zeros(len(logits),next_position-position-1,device=logits.device,dtype=torch.long)),1)
        future_logits=self._reference_anchor[:,next_position] if self._reference_anchor is not None else self.decode_counts(memory,phase,condition,future_history,plan,True)[:,-1]
        future_count=future_logits.argmax(-1)+1
        bonus=next_event_viability(lane_state,current_beat,next_beat,future_count,plan[:,position])
        if bool((bonus>0).any()):self._expression_stats['viability_scored_rows']+=1
        return logits+bonus

    @torch.inference_mode()
    def generate(self,*args,motion_state=None,**kwargs):
        if args:
            bound=inspect.signature(FactorizedRiceGenerator.generate).bind(self,*args,**kwargs)
            kwargs=dict(bound.arguments);kwargs.pop('self');args=()
        phase=kwargs.get('phase',args[2] if len(args)>2 else None)
        plan=kwargs.get('plan',args[5] if len(args)>5 else None)
        beats=kwargs.get('candidate_beats')
        if beats is None:beats=candidate_grid(phase.shape[0],phase.shape[1],phase.device)
        state=motion_state if motion_state is not None else MotionHistory()
        prefix=kwargs.get('prefix_masks')
        self._expression_stats={'independent_music_history':True,'viability_scored_rows':0,'stream_scored_rows':0,'expression_decoding':self.expression_decoding,
            'count_policy':self.count_policy,'count_seed':self.count_seed,'count_temperature':self.count_temperature,
            'reference_counts_outside_hs_js':self.reference_counts is not None,
            'canonical_reference_anchor':False,'protected_count_changes':0,'protected_music_rows':0,
            'music_sampled_rows':0,'music_nonmodal_rows':0,'sampling_log_regret_sum':0.,'density_guarded_rows':0,
            'density_infeasible_rows':0}
        self._music_density=kwargs.get('density')
        self._density_beats=beats
        try:
            if self.reference_counts is not None and kwargs.get('oracle_counts') is None and kwargs.get('row_budgets') is None:
                self._expression_stats['canonical_reference_anchor']=True
                shadow=getattr(state,'expression_shadow',None)
                if shadow is None:
                    from .playstyle_control import PlaystyleControlConfig,PlaystyleLaneControlState,PlaystylePhraseControlState
                    from .lane_control import LaneControlConfig
                    source_motion=MotionHistory()
                    if state.masks is not None:
                        source_motion.masks=state.masks.clone();source_motion.beats=state.beats.clone()
                    source_lane=copy.deepcopy(kwargs.get('lane_state'))
                    source_phrase=copy.deepcopy(kwargs.get('phrase_state'))
                    config=PlaystyleControlConfig(cut_phrase_context=False)
                    if source_lane is None:
                        source_lane=PlaystyleLaneControlState(len(phase),phase.device,LaneControlConfig(policy='style-aware',allow_longjacks=True),config)
                    if source_phrase is None:source_phrase=PlaystylePhraseControlState(len(phase),phase.device,config)
                    if hasattr(source_phrase,'playstyle'):source_phrase.playstyle=replace(source_phrase.playstyle,cut_phrase_context=False)
                    shadow={'motion':source_motion,'lane':source_lane,'phrase':source_phrase}
                    state.expression_shadow=shadow
                source_kwargs=dict(kwargs)
                source_kwargs.update(candidate_beats=beats,lane_state=shadow['lane'],phrase_state=shadow['phrase'])
                if prefix is not None and prefix.shape[1] and shadow['motion'].masks is not None:
                    matches=(shadow['motion'].beats[:,:,None]-beats[:,None,:prefix.shape[1]]).abs()<=1e-7
                    if not matches.any(1).all():raise ValueError('Source musical trajectory prefix is missing.')
                    source_kwargs['prefix_masks']=shadow['motion'].masks.gather(1,matches.long().argmax(1))
                self._reference_pass=True
                reference_result=super().generate(*args,motion_state=shadow['motion'],**source_kwargs)
                self._reference_pass=False
                probability=reference_result['conditional_count_probability']
                prefix_slots=phase.shape[1]-probability.shape[1]
                self._reference_anchor=torch.cat((torch.full((len(phase),prefix_slots,4),.25,device=phase.device),probability),1).clamp_min(1e-12).log()
            if self.count_policy in ('music-sample','music-density-sample','selected-onset-density-sample'):
                if not 0<self.count_temperature<=2:raise ValueError('Musical count sampling needs a temperature in (0,2].')
                generator=getattr(state,'expression_count_generator',None)
                if generator is None:
                    generator=torch.Generator(device=phase.device).manual_seed(self.count_seed)
                    state.expression_count_generator=generator
                self._count_generator=generator
            if prefix is not None and prefix.shape[1]:
                stored=getattr(state,'expression_music',None)
                if stored is not None:
                    matches=(stored[1][:,:,None]-beats[:,None,:prefix.shape[1]]).abs()<=1e-7
                    if not matches.any(1).all():raise ValueError('Independent musical overlap is missing.')
                    self._music_prefix=stored[0].gather(1,matches.long().argmax(1))
            result=super().generate(*args,motion_state=state,**kwargs)
            if self._reference_anchor is not None:
                result['source_conditional_count_probability']=reference_result['conditional_count_probability']
                result['source_music_counts']=reference_result['music_counts']
                prefix_slots=phase.shape[1]-reference_result['conditional_count_probability'].shape[1]
                protected=plan[:,prefix_slots:,[LABELS.index('Jumpstream'),LABELS.index('Handstream')]].amax(-1)<=.5
                changed=(result['music_counts'][:,prefix_slots:]!=reference_result['music_counts'][:,prefix_slots:])&protected
                if getattr(self,'music_chord_support_enabled',False) and self._supported_music_counts is not None:
                    approved=(reference_result['music_counts'][:,prefix_slots:]==1)&(result['music_counts'][:,prefix_slots:]==2)&(self._supported_music_counts[:,prefix_slots:]==2)
                    self._expression_stats['supported_double_restorations']=int((changed&approved).sum())
                    changed=changed&~approved
                if getattr(self,'learned_music_counts',False):
                    learned=plan[:,prefix_slots:,[LABELS.index(s) for s in ('Jacks','Chordjacks','Minijacks','Gluts')]].amax(-1)>.5
                    self._expression_stats['learned_music_count_changed_rows']=int((changed&learned).sum())
                    delta=result['music_counts'][:,prefix_slots:]-reference_result['music_counts'][:,prefix_slots:]
                    self._expression_stats['learned_music_count_net_notes']=int((delta*learned).sum())
                    changed=changed&~learned
                self._expression_stats['protected_count_changes']=int(changed.sum())
                self._expression_stats['protected_music_rows']=int(((result['music_counts'][:,prefix_slots:]>0)&protected).sum())
                if bool(changed.any()):raise ValueError('Protected musical count trajectory changed.')
            state.expression_music=(result['music_counts'].detach().clone(),beats.detach().clone())
            result['phrase_expression']=dict(self._expression_stats)
            return result
        finally:
            self._music_prefix=None;self._expression_stats=None;self._count_generator=None;self._music_density=None
            self._reference_anchor=None;self._reference_pass=False
            self._selected_music_rows=None;self._density_beats=None

    def configuration(self):
        return {**super().configuration(),'architecture':'phrase-expression-v1','layout_residual_bound':1.5,
                'reference_count_fallback':self.reference_counts is not None,
                'reference_layout_fallback':self.reference_layout_decoder is not None,
                'music_history':'independent pre-lane counts; no layout-to-music feedback'}

"""Current-policy count continuation and count-aware Mini soft lookahead."""
import torch
from .music_count_natural import MusicCountNaturalGenerator
from .expression_decoder import next_event_viability
from .segment_labels import LABELS

SCHEMA='generated-musical-count-prefix-v1'


class CountContinuationGenerator(MusicCountNaturalGenerator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.learned_count_lookahead_enabled=False

    def generation_layout_scores(self,logits,memory,phase,condition,previous_counts,desired,
                                 plan,position,candidate_beats,lane_state,onset_probability):
        scores=super().generation_layout_scores(logits,memory,phase,condition,previous_counts,desired,
                                               plan,position,candidate_beats,lane_state,onset_probability)
        if (not self.learned_count_lookahead_enabled or self._reference_pass or not self.expression_decoding
            or self._reference_anchor is None or lane_state is None or not hasattr(lane_state,'sixteenth_runs')
            or not bool((desired>0).any()) or not bool((plan[:,position,LABELS.index('Minijacks')]>.5).any())):
            return scores
        future=(onset_probability[0,position+1:]>=.5).nonzero().flatten()
        if not len(future):return scores
        next_position=position+1+int(future[0]);current=candidate_beats[:,position];next_beat=candidate_beats[:,next_position]
        if not bool((next_beat-current<=.5+1e-7).all()) or not torch.equal(plan[:,next_position],plan[:,position]):return scores
        # Counts are musical desired counts, before lane reductions. No candidate
        # mask/hand pressure enters this next-count prediction.
        history=torch.cat((previous_counts,desired[:,None],torch.zeros(len(logits),next_position-position-1,device=logits.device,dtype=torch.long)),1)
        future_logits=self.generation_count_logits(memory,phase,condition,history,None,plan,True)[:,-1]
        old=self._reference_anchor[:,next_position].argmax(-1)+1
        new=future_logits.argmax(-1)+1
        old_bonus=next_event_viability(lane_state,current,next_beat,old,plan[:,position])
        new_bonus=next_event_viability(lane_state,current,next_beat,new,plan[:,position])
        if self._expression_stats is not None:
            self._expression_stats['learned_count_lookahead_changed_rows']=self._expression_stats.get('learned_count_lookahead_changed_rows',0)+int((new!=old).sum())
        # Replace only the old soft bonus. Never change the count or legal mask.
        return scores-old_bonus+new_bonus

    def configuration(self):
        return {**super().configuration(),'architecture':'count-context-v1',
                'generated_count_prefix_schema':SCHEMA,'mini_count_lookahead_schema':'learned-next-count-soft-v1'}

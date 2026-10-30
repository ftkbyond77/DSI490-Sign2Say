import unittest
from pathlib import Path
import numpy as np
from manual_features import detailed_profile
from reranker import load_manual,manual_match,rerank_detailed
class ManualTests(unittest.TestCase):
    def setUp(self):
        self.v=dict(id='v',num_hands=2,bimanual_type='symmetric',active_hand='both',wrist_y_min=-.5,wrist_y_max=.8,target_distance_lr_wrist_max=.5,checks={},source={'pdf_page':1})
        self.q=dict(observed_rate=[1,1],wrist_y=[.2,.2],median_speed=[1,1],distance_lr_wrist=.2,circling=[1,1],speed_balance=1,alternating=1)
    def test_missing_hand_abstains(self):
        self.q['observed_rate'][1]=.05;self.q['distance_lr_wrist']=9
        r=manual_match(self.q,{'variants':[self.v]});self.assertEqual(r['rule_bonus'],0);self.assertEqual(r['best']['coverage'],0)
    def test_symmetric_not_implicitly_close(self):
        self.v['target_distance_lr_wrist_max']=None;self.q['distance_lr_wrist']=9
        r=manual_match(self.q,{'variants':[self.v]});self.assertGreater(r['rule_bonus'],0);self.assertNotIn('wrist_distance',r['best']['parts'])
    def test_far_hands_penalty_is_soft(self):
        self.q['distance_lr_wrist']=9;self.q['wrist_y']=[9,9]
        r=manual_match(self.q,{'variants':[self.v]});self.assertAlmostEqual(r['rule_bonus'],-.2)
    def test_variant_accepts_alternative(self):
        self.q['distance_lr_wrist']=9
        alternative=dict(self.v,id='alternative',target_distance_lr_wrist_max=None)
        r=manual_match(self.q,{'variants':[self.v,alternative]});self.assertEqual(r['best']['variant'],'alternative')
    def test_unknown_word_unchanged_and_bounded(self):
        cs=[dict(word='unlisted',support=.5),dict(word='listed',support=.3)]
        r=rerank_detailed(cs,dict(observed_rate=[1,1]),{}, {'listed':{'variants':[self.v]}}, self.q)
        d={x['word']:x for x in r};self.assertEqual(d['unlisted']['final_score'],.5)
        self.assertLessEqual(d['listed']['bonus'],.15);self.assertIsNone(d['listed']['p_correct'])
    def test_circle_vs_line(self):
        t=np.arange(0,2,.04);phase=t/2*2*np.pi;x=np.stack([.25*np.cos(phase),.25*np.sin(phase)],1)
        s=dict(t=t,xy=np.stack([x,x],1),observed=np.ones((len(t),2),bool))
        r=detailed_profile(s,0,2);self.assertGreater(r['circling'][0],.7)
        s['xy'][:,:,0]=t[:,None];s['xy'][:,:,1]=0
        r=detailed_profile(s,0,2);self.assertLess(r['circling'][0],.1)
    def test_gaps_do_not_make_circle(self):
        t=np.arange(0,2,.04);p=t*np.pi;x=np.stack([.2*np.cos(p),.2*np.sin(p)],1)
        o=np.ones((len(t),2),bool);o[::3]=False
        r=detailed_profile(dict(t=t,xy=np.stack([x,x],1),observed=o),0,2)
        self.assertIsNone(r['circling'][0])
    def test_load_external_schema(self):
        d=load_manual(Path(__file__).with_name('manual_heuristics_tsl_detailed.json'));self.assertEqual(len(d),6)
if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from segment_merger import assess
from reranker import rerank
class Tests(unittest.TestCase):
    def setUp(self):
        t=np.arange(0,2,.04);x=np.zeros((len(t),2,2));x[:,:,0]=t[:,None]
        self.s=dict(t=t,xy=x,observed=np.ones((len(t),2),bool),face=np.zeros(len(t)))
        self.a=dict(t0=0.,t1=1.,top5=['a','b','c','d','e']);self.b=dict(t0=1.,t1=2.,top5=['c','f','g','h','i'])
    def test_continuity(self):self.assertTrue(assess(self.a,self.b,self.s)['merge'])
    def test_gap(self):self.b['t0']=1.2;self.assertFalse(assess(self.a,self.b,self.s)['merge'])
    def test_top5_only_overlap(self):self.b['top5']=['f','g','h','c'];self.assertFalse(assess(self.a,self.b,self.s)['merge'])
    def test_duration(self):self.b['t1']=3.21;self.assertFalse(assess(self.a,self.b,self.s)['merge'])
    def test_disappearance(self):self.s['observed'][25,1]=False;self.assertFalse(assess(self.a,self.b,self.s)['merge'])
    def test_rest(self):self.s['xy'][:]=0;self.assertFalse(assess(self.a,self.b,self.s)['merge'])
    def test_jump(self):self.s['xy'][25:,:,0]+=1;self.assertFalse(assess(self.a,self.b,self.s)['merge'])
    def test_missing_masks_rules(self):
        q=dict(observed_rate=[1,.05],is_two_handed=0,symmetry=0)
        table={'a':dict(is_two_handed_median=1,is_two_handed_count=10,is_two_handed_std=.1)}
        r=rerank([dict(word='a',support=.3)],q,table)[0];self.assertEqual(r['bonus'],0);self.assertIsNone(r['p_correct'])
    def test_unknown_reference(self):self.assertEqual(rerank([dict(word='a',support=.3)],dict(observed_rate=[1,1]),{})[0]['final_score'],.3)
if __name__=='__main__':unittest.main()

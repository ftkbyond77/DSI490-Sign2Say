import unittest
import numpy as np
from merge_rule import assess,MergeConfig
class GateTests(unittest.TestCase):
 def setUp(self):
  self.t=np.arange(100)/25;self.xy=np.zeros((100,2,21,2));self.xy[:,:,:,0]=self.t[:,None,None]*.5;self.obs=np.ones((100,2),bool)
  self.a=dict(t0=.2,t1=1.2,top5=['a','b','c','d','e']);self.b=dict(t0=1.2,t1=2.4,top5=['a','b','f','g','h']);self.cfg=MergeConfig(min_continuous_hands=2)
 def result(self):return assess(self.a,self.b,self.t,self.xy,self.obs,self.cfg)['merge']
 def test_continuous_positive(self):self.assertTrue(self.result())
 def test_gap_equal_point_two_rejected(self):self.b['t0']=1.4;self.assertFalse(self.result())
 def test_disjoint_candidates_rejected(self):self.b['top5']=['v','w','x','y','z'];self.assertFalse(self.result())
 def test_pause_rejected(self):self.xy[:]=0.;self.assertFalse(self.result())
 def test_jump_rejected(self):self.xy[self.t>=1.2]+=5.;self.assertFalse(self.result())
 def test_missing_hand_abstains(self):self.obs[:,1]=False;self.assertFalse(self.result())
 def test_overlapping_time_rejected(self):self.b['t0']=1.1;self.assertFalse(self.result())
 def test_long_combination_rejected(self):self.b['t1']=3.8;self.assertFalse(self.result())
if __name__=='__main__':unittest.main()

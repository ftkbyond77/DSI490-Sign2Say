import unittest
import numpy as np
from gesture_detector import hand_features,detect_gestures

def raw_fixture():
 n=20;p=np.zeros((n,33,4));p[:,11]=[.3,.4,0,1];p[:,12]=[.7,.4,0,1]
 hand=np.zeros((21,3));hand[0]=[.43,.5,0]
 for base,x in [(5,.40),(9,.42),(13,.44),(17,.46)]:
  for j in range(4):hand[base+j]=[x,.46-j*.025,0]
 hand[1:5]=[[.39,.49,0],[.38,.47,0],[.37,.45,0],[.36,.43,0]]
 other=hand.copy();other[:,0]+=.12
 return dict(t=np.arange(n)*.04,pose=p,lh=np.tile(hand,(n,1,1)),rh=np.tile(other,(n,1,1)),hw=(600,600))
class GestureTests(unittest.TestCase):
 def test_extended(self):self.assertTrue(hand_features(raw_fixture()['lh'][0,:,:2])['four_fingers_extended'])
 def test_prayer_like(self):self.assertEqual(detect_gestures(raw_fixture(),0,1)['labels']['prayer_like_2d']['status'],'supported')
 def test_missing_not_negative(self):
  r=raw_fixture();r['rh'][:]=0;self.assertEqual(detect_gestures(r,0,1)['labels']['prayer_like_2d']['status'],'insufficient_data')
 def test_close_not_enough(self):
  r=raw_fixture()
  for h in ['lh','rh']:
   for b in [5,9,13,17]:r[h][:,b+2,1]=.48;r[h][:,b+3,1]=.49
  self.assertNotEqual(detect_gestures(r,0,1)['labels']['prayer_like_2d']['status'],'supported')
 def test_far_hands(self):
  r=raw_fixture();r['rh'][:,:,0]+=.4;self.assertEqual(detect_gestures(r,0,1)['labels']['prayer_like_2d']['status'],'not_supported')
 def test_single_frame_not_persistent(self):self.assertNotEqual(detect_gestures(raw_fixture(),0,.04)['labels']['prayer_like_2d']['status'],'supported')
 def test_degenerate(self):self.assertIsNone(hand_features(np.zeros((21,2))))
 def test_nonfinite(self):
  p=raw_fixture()['lh'][0,:,:2];p[8,0]=np.nan;self.assertIsNone(hand_features(p))
 def test_no_contact_claim(self):self.assertEqual(hand_features(raw_fixture()['lh'][0,:,:2])['contact'],'unknown_2d')
 def test_translation_scale_invariance(self):
  p=raw_fixture()['lh'][0,:,:2];self.assertEqual(hand_features(p)['fingers'],hand_features(p*3+[2,1])['fingers'])
if __name__=='__main__':unittest.main()

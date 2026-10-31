import unittest
import torch
from modules.masked_hand import HandReconstruction, mask_hands, reconstruction_loss

class MaskedHandTests(unittest.TestCase):
    def data(self):
        torch.manual_seed(0)
        parts = {h:torch.rand(2, 6, 21, 3) for h in ('left','right')}
        for x in parts.values(): x[...,2] = 1
        valid = torch.ones(2,6,dtype=torch.bool); valid[1,4:] = False
        observed = {h:torch.ones(2,6,21,dtype=torch.bool) for h in parts}
        observed['left'][0,1] = False
        return parts,valid,observed

    def test_mask_excludes_imputed_and_padding(self):
        p,v,o = self.data()
        x,t,m = mask_hands(p,v,o,probability=1)
        self.assertFalse(m[0,1,:21].any())
        self.assertFalse(m[1,4:].any())
        self.assertTrue(torch.equal(t[:,:,:21],p['left'][...,:2]))
        self.assertEqual(float(x['left'][v].abs().sum()),0)
        self.assertTrue(torch.all(p['left'][...,2] == 1))

    def test_missing_wrist_invalidates_relative_targets(self):
        p,v,o = self.data(); o['right'][0,2,0] = False
        _,_,m = mask_hands(p,v,o,1)
        self.assertFalse(m[0,2,21:].any())

    def test_unobserved_values_do_not_affect_loss(self):
        p,v,o = self.data(); _,t,m = mask_hands(p,v,o,1)
        pred = torch.zeros_like(t,requires_grad=True)
        a = reconstruction_loss(pred,t,m)
        changed = t.clone(); changed[~m] = 1e6
        b = reconstruction_loss(pred,changed,m)
        self.assertEqual(float(a.detach()),float(b.detach()))
        a.backward(); self.assertEqual(float(pred.grad[~m].abs().sum()),0)

    def test_empty_mask_has_finite_zero_gradient(self):
        pred = torch.randn(2,3,42,2,requires_grad=True)
        loss = reconstruction_loss(pred,torch.zeros_like(pred),torch.zeros(2,3,42,dtype=torch.bool))
        loss.backward()
        self.assertEqual(float(loss.detach()),0)
        self.assertEqual(float(pred.grad.abs().sum()),0)

    def test_missing_provenance_and_wrong_time_rejected(self):
        p,v,o = self.data()
        with self.assertRaises(KeyError): mask_hands(p,v,{})
        with self.assertRaises(ValueError): reconstruction_loss(torch.zeros(2,2,42,2),torch.zeros(2,3,42,2),torch.zeros(2,3,42,dtype=torch.bool))

    def test_auxiliary_gradient_reaches_features(self):
        h = HandReconstruction(16)
        features = torch.randn(2,6,16,requires_grad=True)
        target = torch.randn(2,6,42,2)
        reconstruction_loss(h(features),target,torch.ones(2,6,42,dtype=torch.bool)).backward()
        self.assertGreater(float(features.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is not None for p in h.parameters()))

if __name__ == '__main__': unittest.main()

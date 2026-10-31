"""Protocol regression tests, independent numerical oracles and gradient checks."""
import unittest
import torch
import torch.nn.functional as F
from zsslr import ZSSLRCompatibilityModel, training_loss, predict, evaluate, FlipDifferenceOperator
from zsslr.protocol import calibrate,local_targets

class Tests(unittest.TestCase):
    """Verify zero-shot restrictions and counterfactual math."""
    def setUp(self) -> None:
        """Small deterministic data with noncontiguous global class IDs."""
        torch.manual_seed(3); torch.set_num_threads(2)
        self.m=ZSSLRCompatibilityModel(width=4)
        self.b=torch.randn(2,3,3,16,16)
        self.t=torch.randn(4,768); self.a=torch.randint(0,2,(4,53)).float()
        self.s=torch.tensor([2,0]); self.u=torch.tensor([3,1])

    def test_training(self) -> None:
        """CE uses reordered seen columns and gradients reach both streams."""
        y=torch.tensor([0,2])
        out=training_loss(self.m,self.b,self.b,self.t,self.a,y,self.s,.02)
        expected=F.cross_entropy(self.m(self.b,self.b,self.t[self.s],self.a[self.s]),torch.tensor([1,0]))
        self.assertTrue(torch.allclose(out['loss'],expected+.02*self.m.W.square().sum()))
        out['loss'].backward()
        for module in (self.m.visual.body,self.m.visual.hand,self.m.semantic):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
        changed=self.t.clone(); changed[self.u]+=999
        other=training_loss(self.m,self.b,self.b,changed,self.a,y,self.s,.02)
        self.assertTrue(torch.allclose(out['loss'],other['loss']))
        with self.assertRaises(ValueError): local_targets(torch.tensor([1]),self.s)

    def test_modes(self) -> None:
        """Candidate restriction and gamma subtraction are correct by global IDs."""
        z,ids=predict(self.m,self.b,self.b,self.t,self.a,self.s,self.u)
        self.assertEqual(ids.tolist(),[3,1]); self.assertEqual(tuple(z.shape),(2,2))
        g,ids=predict(self.m,self.b,self.b,self.t,self.a,self.s,self.u,'gzsslr',.7)
        raw=self.m(self.b,self.b,self.t[ids],self.a[ids])
        self.assertTrue(torch.allclose(g[:,:2],raw[:,:2]-.7))
        self.assertTrue(torch.allclose(g[:,2:],raw[:,2:]))
        with self.assertRaises(ValueError): predict(self.m,self.b,self.b,self.t,self.a,self.s,self.s)

    def test_metrics(self) -> None:
        """Known macro S=1 U=0.75 implies H=6/7."""
        logits=torch.tensor([[4.,0,0],[0,4.,0],[4.,0,0],[0,0,4.]])
        result=evaluate(logits,torch.tensor([0,1,1,2]),torch.arange(3),torch.tensor([0]))
        self.assertAlmostEqual(result['H'],6/7); self.assertEqual(result['top5'],1.)

    def test_flip(self) -> None:
        """Compare one flip against full recomputation; inputs stay unchanged."""
        self.m.eval()
        with torch.no_grad(): v=self.m.visual(self.b,self.b)
        orig=self.a.clone(); ids=torch.arange(4)
        result=FlipDifferenceOperator(self.m,v,self.t,self.a,ids,self.s,.3)
        a=self.a.clone(); a[2,7]=1-a[2,7]
        expected=calibrate(self.m.score_features(v,self.t,a),ids,self.s,.3).softmax(-1)-result['baseline_probability']
        self.assertTrue(torch.allclose(expected,result['probability_shift'][:,2,7,:],atol=1e-7))
        self.assertTrue(torch.equal(orig,self.a))
        self.assertTrue(torch.allclose(result['probability_shift'].sum(-1),torch.zeros(2,4,53),atol=1e-6))

    def test_invalid_attributes(self) -> None:
        """Unknown attributes cannot silently become negative/absent labels."""
        a=self.a.clone(); a[0,0]=float('nan')
        with self.assertRaises(ValueError): self.m.semantic(self.t,a)

if __name__=='__main__': unittest.main()

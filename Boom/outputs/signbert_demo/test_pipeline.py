"""CPU tests of the encoder and direct coordinate reconstruction."""
import unittest
import torch
from model import HandTransformer,CoordinateDecoder,masked_input,reconstruction_loss

class PipelineTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(42); torch.set_num_threads(2)
    def test_encoder_backprop(self):
        model=HandTransformer(5,dim=32,heads=4,layers=1,frames=8)
        x=torch.randn(2,8,2,21,2); mask=torch.ones(2,8,2,21,dtype=torch.bool)
        output=model(x,mask)
        self.assertEqual(output.shape,(2,5))
        torch.nn.functional.cross_entropy(output,torch.tensor([1,3])).backward()
        self.assertTrue(torch.isfinite(model.graph[0].linear.weight.grad).all())
    def test_no_hand_rejected(self):
        model=HandTransformer(2,dim=32,layers=1,frames=8)
        with self.assertRaisesRegex(ValueError,'No observed'):
            model(torch.zeros(1,8,2,21,2),torch.zeros(1,8,2,21,dtype=torch.bool))
    def test_mask_loss_ignores_missing(self):
        x=torch.randn(2,8,2,21,2); mask=torch.ones(2,8,2,21,dtype=torch.bool)
        mask[:,:,0]=False
        corrupted,observed,target,loss_mask=masked_input(x,mask)
        self.assertFalse(loss_mask[:,:,0].any())
        self.assertTrue(loss_mask.any())
        altered=target.clone(); altered[:,:,0]=9999
        self.assertEqual(float(reconstruction_loss(altered,target,loss_mask)),0.)
    def test_coordinate_decoder_backprop(self):
        model=HandTransformer(5,dim=32,heads=4,layers=1,frames=8)
        decoder=CoordinateDecoder(32)
        x=torch.randn(2,8,2,21,2); mask=torch.ones(2,8,2,21,dtype=torch.bool)
        corrupted,observed,target,lmask=masked_input(x,mask)
        prediction=decoder(model.encode(corrupted,observed))
        self.assertEqual(prediction.shape,x.shape)
        loss=reconstruction_loss(prediction,target,lmask); loss.backward()
        self.assertTrue(torch.isfinite(decoder.net[-1].weight.grad).all())
        self.assertTrue(torch.isfinite(model.graph[0].linear.weight.grad).all())
    def test_no_reconstruction_evidence_rejected(self):
        with self.assertRaisesRegex(ValueError,'No observed joints'):
            reconstruction_loss(torch.zeros(1,21,2),torch.zeros(1,21,2),torch.zeros(1,21,dtype=torch.bool))

if __name__=='__main__': unittest.main()

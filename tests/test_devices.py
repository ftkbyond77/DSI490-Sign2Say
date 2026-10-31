"""Device policy regression tests; no model downloads or cloud access."""
import os
import unittest
from unittest.mock import patch

import torch
from modules.utils import device, inference_precision


class DevicePolicyTests(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        for cuda, mps, expected in [(True, True, 'cuda'), (False, True, 'mps'), (False, False, 'cpu')]:
            with patch('torch.cuda.is_available', return_value=cuda), patch('torch.backends.mps.is_available', return_value=mps):
                self.assertEqual(device('auto'), expected)

    def test_cpu_override_and_environment(self):
        with patch.dict(os.environ, {'DEVICE': 'cpu'}):
            self.assertEqual(device(), 'cpu')
            self.assertEqual(device(torch.device('cpu')), 'cpu')

    def test_unavailable_accelerator_is_explicit(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                device('cuda')
        with patch('torch.backends.mps.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                device('mps')

    def test_invalid_cuda_index(self):
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=1):
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                device('cuda:2')

    def test_cpu_and_mps_do_not_enter_cuda_amp(self):
        with patch('torch.autocast', side_effect=AssertionError('Unexpected AMP')):
            for target in ['cpu', 'mps']:
                with inference_precision(target):
                    pass

    def test_cpu_training_step(self):
        net = torch.nn.Linear(4, 2).to(device('cpu'))
        opt = torch.optim.AdamW(net.parameters(), lr=0.01)
        before = net.weight.detach().clone()
        with inference_precision('cpu'):
            loss = net(torch.ones(2, 4)).square().mean()
        loss.backward()
        opt.step()
        self.assertFalse(torch.equal(before, net.weight))


if __name__ == '__main__':
    unittest.main()

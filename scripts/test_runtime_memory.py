import unittest
from types import SimpleNamespace
from unittest.mock import patch
from training.execution import trim_cuda_cache
from scripts.teaching_supervisor import controller_staleness

class CachePressureTests(unittest.TestCase):
    def test_long_hashing_with_cpu_progress_is_not_a_dead_controller(self):
        state,dead=controller_staleness({},'a',500)
        self.assertFalse(dead)
        state,dead=controller_staleness(state,'b',800)
        self.assertFalse(dead)
        state,dead=controller_staleness(state,'b',1100)
        self.assertFalse(dead)
        state,dead=controller_staleness(state,'b',1400)
        self.assertTrue(dead)
        self.assertFalse(controller_staleness(state,'b',10)[1])
    def test_unused_cache_pressure_released_but_live_memory_preserved(self):
        with patch('training.execution.torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=16000)), patch('training.execution.torch.cuda.memory_reserved',return_value=11000), patch('training.execution.torch.cuda.memory_allocated',return_value=800), patch('training.execution.torch.cuda.empty_cache') as clear:
            self.assertTrue(trim_cuda_cache('cuda:0'))
            clear.assert_called_once()
        with patch('training.execution.torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=16000)), patch('training.execution.torch.cuda.memory_reserved',return_value=11000), patch('training.execution.torch.cuda.memory_allocated',return_value=10000), patch('training.execution.torch.cuda.empty_cache') as clear:
            self.assertFalse(trim_cuda_cache('cuda:0'))
            clear.assert_not_called()
        with patch('training.execution.torch.cuda.get_device_properties') as cuda:
            self.assertFalse(trim_cuda_cache('cpu'))
            cuda.assert_not_called()

if __name__=='__main__':unittest.main()

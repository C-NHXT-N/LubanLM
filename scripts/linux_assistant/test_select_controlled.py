import unittest
from scripts.linux_assistant.select_controlled import eligible,paired_bootstrap,wilson


class ControlledSelectionTests(unittest.TestCase):
    def test_wilson_and_paired_delta(self):
        low,high=wilson(80,100)
        self.assertLess(low,.8);self.assertGreater(high,.8)
        self.assertEqual(paired_bootstrap([0,0,1,1],[1,1,1,1],iterations=1000),[0.0,1.0])

    def test_eligibility_requires_each_task_slice_and_both_general_metrics(self):
        metric=lambda rate:{"rate":rate}
        base={"task":metric(.60),"command_generation":metric(.60),"log_troubleshooting":metric(.60),
              "general_full":metric(.90),"general_final":metric(.95)}
        candidate={"task":metric(.82),"command_generation":metric(.82),"log_troubleshooting":metric(.80),
                   "general_full":metric(.86),"general_final":metric(.90)}
        thresholds={"task_min":.8,"command_min":.8,"troubleshooting_min":.8,"task_gain":.1,
                    "error_reduction":.3,"general_max_drop":.05}
        self.assertTrue(eligible(base,candidate,thresholds))
        candidate["general_final"]={"rate":.89}
        self.assertFalse(eligible(base,candidate,thresholds))


if __name__=="__main__":unittest.main()

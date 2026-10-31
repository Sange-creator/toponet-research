"""Preserve covered numerical behavior while moving canonical code into a package."""

import unittest


class CoreTests(unittest.TestCase):
    def test_network(self):
        from toponet.models.graph_q import self_test

        self_test()

    def test_double_dqn(self):
        from toponet.agents.ddqn import self_test

        self_test()

    def test_environment(self):
        from toponet.envs.handover import self_test

        self_test()

    def test_features_rewards(self):
        from toponet.envs.features import self_test

        self_test()

    def test_radio(self):
        from toponet.envs.radio import self_test

        self_test()

    def test_interval_service(self):
        from toponet.envs.service import self_test

        self_test()

    def test_allocator(self):
        from toponet.envs.allocation import self_test

        self_test()

    def test_traffic(self):
        from toponet.envs.traffic import self_test

        self_test()

    def test_baselines(self):
        from toponet.baselines.controllers import self_test

        self_test()


if __name__ == "__main__":
    unittest.main(verbosity=2)

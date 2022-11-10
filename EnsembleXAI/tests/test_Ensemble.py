import torch as t

from EnsembleXAI import Ensemble
from unittest import TestCase


class TestAggregate(TestCase):
    exp1 = t.ones([2, 2])
    exp3 = 3 * t.ones([2, 2])
    obs1_tensor = t.stack((exp1, exp3))
    exp0 = t.zeros([2, 2])
    obs2_tensor = t.stack([exp0, exp3])

    mult_obs_tensor = t.stack([obs1_tensor, obs2_tensor])

    def test_one_obs_avg(self):
        # tuple input
        ensembled = Ensemble.aggregate((self.exp1, self.exp3), 'avg')
        self.assertIsInstance(ensembled, t.Tensor)

        expected = 2 * t.ones([1, 2, 2])
        self.assertTrue(t.equal(ensembled, expected))
        # tensor input
        ensembled = Ensemble.aggregate(self.obs1_tensor, 'avg')
        self.assertTrue(t.equal(ensembled, expected))

    def test_one_obs_max(self):
        ensembled = Ensemble.aggregate((self.exp1, self.exp3), 'max')
        self.assertIsInstance(ensembled, t.Tensor)

        expected = 3 * t.ones([1, 2, 2])
        self.assertTrue(t.equal(ensembled, expected))
        # tensor input
        ensembled = Ensemble.aggregate(self.obs1_tensor, 'max')
        self.assertTrue(t.equal(ensembled, expected))

    def test_one_obs_min(self):
        ensembled = Ensemble.aggregate((self.exp1, self.exp3), 'min')
        self.assertIsInstance(ensembled, t.Tensor)

        expected = t.ones([1, 2, 2])
        self.assertTrue(t.equal(ensembled, expected))
        # tensor input
        ensembled = Ensemble.aggregate(self.obs1_tensor, 'min')
        self.assertTrue(t.equal(ensembled, expected))

    def test_multi_obs_avg(self):
        # tuple input
        ensembled = Ensemble.aggregate((self.obs1_tensor, self.obs2_tensor), 'avg')
        self.assertIsInstance(ensembled, t.Tensor)
        expected = t.stack([2 * t.ones([2, 2]), 1.5 * t.ones([2, 2])])
        self.assertTrue(t.equal(ensembled, expected))

        # tensor input
        ensembled = Ensemble.aggregate(self.mult_obs_tensor, 'avg')
        self.assertTrue(t.equal(ensembled, expected))

    def test_illegal_args(self):
        with self.assertRaises(AssertionError):
            Ensemble.aggregate(self.obs1_tensor, 'asdf')
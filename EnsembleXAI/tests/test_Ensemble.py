import torch as t

from EnsembleXAI import Ensemble
from unittest import TestCase

from EnsembleXAI.Ensemble import _normalize_across_dataset, _reformat_input_tensors


def _dummy_metric(x: t.Tensor) -> t.Tensor:
    return t.ones([x.size(dim=0)])


def _dummy_metric2(x: t.Tensor) -> t.Tensor:
    return 2 * _dummy_metric(x)


class TestReformatInputs(TestCase):
    def test_tensors(self):
        actual = _reformat_input_tensors(t.zeros([1, 3, 32, 32]))
        expected = t.zeros((1, 1, 3, 32, 32))
        self.assertTrue(t.equal(actual, expected))

        actual = _reformat_input_tensors(t.zeros([3, 1, 32, 32]))
        expected = _reformat_input_tensors(t.zeros([1, 3, 1, 32, 32]))
        self.assertTrue(t.equal(actual, expected))

    def test_tuples(self):

        x = t.tensor([[[[0, 1], [1, 0]], [[0, 1], [1, 0]]],
                       [[[0, 1], [1, 0]], [[0, 1], [1, 0]]]], dtype=t.float)
        y = x

        actual = _reformat_input_tensors((x, y))
        expected = t.tensor([[[[[0, 1], [1, 0]], [[0, 1], [1, 0]]],
                              [[[0, 1], [1, 0]], [[0, 1], [1, 0]]]],
                             [[[[0, 1], [1, 0]], [[0, 1], [1, 0]]],
                              [[[0, 1], [1, 0]], [[0, 1], [1, 0]]]]
                             ], dtype=t.float)
        self.assertTrue(t.equal(actual, expected))


class TestEnsembleXAI(TestCase):
    def test_ensemble_mult_channels_mult_obs(self):
        inputs = t.rand([90, 3, 3, 32, 32])
        masks = t.randint(low=0, high=2, size=[90, 32, 32])
        ensembled = Ensemble.ensembleXAI(inputs, masks, shuffle=False)
        self.assertTrue(ensembled.shape == (90, 3, 32, 32))
        # hard to predict outcome of this algorithm to check exact correctness, even on not random data
        # for now testing only result's shape

    def test_ensemble_one_channel_mult_obs(self):
        inputs = t.rand([90, 3, 1, 32, 32])
        masks = t.randint(low=0, high=2, size=[90, 32, 32])
        ensembled = Ensemble.ensembleXAI(inputs, masks, shuffle=False)
        self.assertTrue(ensembled.shape == (90, 1, 32, 32))
        # hard to predict outcome of this algorithm to check exact correctness, even on not random data
        # for now testing only result's shape

    def test_ensemble_one_channel_one_obs(self):
        inputs = t.rand([3, 1, 32, 32])
        masks = t.randint(low=0, high=2, size=[1, 32, 32])
        with self.assertRaises(AssertionError):
            Ensemble.ensembleXAI(inputs, masks, shuffle=False)
            Ensemble.ensembleXAI(inputs, masks, shuffle=False, n_folds=1)


class TestNormalize(TestCase):
    def test_normalization(self):
        x = t.tensor([[[[[1, 2], [2, 1]]], [[[3, 4], [3, 4]]]],
                      [[[[3, 5], [5, 3]]], [[[0, 1], [1, 0]]]]], dtype=t.float64)
        normalized = _normalize_across_dataset(x)
        expected = t.tensor([[[[[-1.1068, 0.0000],
                                [-0.4743, -0.5916]]],

                              [[[0.1581, 1.1832],
                                [0.1581, 1.1832]]]],

                             [[[[0.1581, 1.7748],
                                [1.4230, 0.5916]]],

                              [[[-1.7393, -0.5916],
                                [-1.1068, -1.1832]]]]], dtype=t.float64)
        self.assertTrue(t.allclose(normalized, expected, atol=0.001))


class TestEnsemble(TestCase):
    x = t.tensor([[[[[0, 1], [1, 0]], [[0, 1], [1, 0]]],
                   [[[0, 1], [1, 0]], [[0, 1], [1, 0]]]]], dtype=t.float)
    y = t.squeeze(x, 0)

    def test_ensemble_single_obs_single_metric_mult_channel(self):
        ensemble = Ensemble.ensemble(self.x, [_dummy_metric], [1])
        self.assertIsInstance(ensemble, t.Tensor)

        expected = t.tensor([[[[-0.9354, 0.9354],
                              [0.9354, -0.9354]],
                             [[-0.9354, 0.9354],
                              [0.9354, -0.9354]]]])

        self.assertTrue(t.allclose(ensemble, expected, atol=.01))

    def test_ensemble_mult_obs_mult_metric_single_channel(self):
        ensemble = Ensemble.ensemble(self.x, [_dummy_metric, _dummy_metric], [0.5, 0.5])
        self.assertIsInstance(ensemble, t.Tensor)

        expected = t.tensor([[[-0.9354, 0.9354],
                              [0.9354, -0.9354]],
                             [[-0.9354, 0.9354],
                              [0.9354, -0.9354]]])

        self.assertTrue(t.allclose(ensemble, expected, atol=.01))

    def test_ensemble_one_obs_one_channel_one_metric(self):
        exp1 = t.tensor([[[[0, 1], [1, 0]]], [[[0, 1], [1, 0]]]], dtype=t.float)
        ensemble = Ensemble.ensemble(tuple(exp1), [_dummy_metric], [1])
        expected = t.tensor([[[[[-0.8660, 0.8660],
                               [0.8660, -0.8660]]]]])
        self.assertTrue(t.allclose(ensemble, expected, atol=.01))

    def test_ensemble_multiple_obs_multiple_channel_single_metric(self):
        ensemble = Ensemble.ensemble((self.y, self.y), [_dummy_metric], [1])
        expected = t.tensor([[[[-0.9682, 0.9682],
                               [0.9682, -0.9682]],
                              [[-0.9682, 0.9682],
                               [0.9682, -0.9682]]],
                             [[[-0.9682, 0.9682],
                               [0.9682, -0.9682]],
                              [[-0.9682, 0.9682],
                               [0.9682, -0.9682]]]
                             ])

        self.assertTrue(t.allclose(ensemble, expected, atol=.01))


class TestAggregate(TestCase):
    exp1 = t.ones([1, 2, 2])
    exp3 = 3 * t.ones([1, 2, 2])
    obs1_tensor = t.stack((exp1, exp3))
    exp0 = t.zeros([1, 2, 2])
    obs2_tensor = t.stack([exp0, exp3])

    obs3_tensor = t.stack((exp1, exp0, exp3)).squeeze()
    mult_obs_tensor = t.stack([obs1_tensor, obs2_tensor])

    def test_one_obs_mult_channels(self):
        ensembled = Ensemble.aggregate(self.obs3_tensor, 'avg')
        self.assertIsInstance(ensembled, t.Tensor)

        self.assertTrue(t.equal(ensembled, self.obs3_tensor[None, :]))

    def test_one_obs_one_channel_avg(self):
        # tuple input
        ensembled = Ensemble.aggregate((self.exp1, self.exp3), 'avg')
        self.assertIsInstance(ensembled, t.Tensor)

        expected = 2 * t.ones([1, 1, 2, 2])
        self.assertTrue(t.equal(ensembled, expected))
        # tensor input
        ensembled = Ensemble.aggregate(self.obs1_tensor, 'avg')
        self.assertTrue(t.equal(ensembled, expected))

    def test_one_obs_one_channel_max(self):
        ensembled = Ensemble.aggregate((self.exp1, self.exp3), 'max')
        self.assertIsInstance(ensembled, t.Tensor)

        expected = 3 * t.ones([1, 1, 2, 2])
        self.assertTrue(t.equal(ensembled, expected))
        # tensor input
        ensembled = Ensemble.aggregate(self.obs1_tensor, 'max')
        self.assertTrue(t.equal(ensembled, expected))

    def test_one_obs_min(self):
        ensembled = Ensemble.aggregate((self.exp1, self.exp3), 'min')
        self.assertIsInstance(ensembled, t.Tensor)

        expected = t.ones([1, 1, 2, 2])
        self.assertTrue(t.equal(ensembled, expected))
        # tensor input
        ensembled = Ensemble.aggregate(self.obs1_tensor, 'min')
        self.assertTrue(t.equal(ensembled, expected))

    def test_multi_obs_avg(self):
        # tuple input
        ensembled = Ensemble.aggregate((self.obs1_tensor, self.obs2_tensor), 'avg')
        self.assertIsInstance(ensembled, t.Tensor)
        expected = t.stack([2 * t.ones([1, 2, 2]), 1.5 * t.ones([1, 2, 2])])
        self.assertTrue(t.equal(ensembled, expected))

        # tensor input
        ensembled = Ensemble.aggregate(self.mult_obs_tensor, 'avg')
        self.assertTrue(t.equal(ensembled, expected))

    def test_max_abs_aggregation(self):
        ensembled = Ensemble.aggregate(self.obs1_tensor, 'max_abs')
        self.assertIsInstance(ensembled, t.Tensor)
        expected = self.exp3.unsqueeze(0)
        self.assertTrue(t.equal(ensembled, expected))

    def test_illegal_args(self):
        with self.assertRaises(AssertionError):
            Ensemble.aggregate(self.obs1_tensor, 'asdf')
            Ensemble.aggregate(self.obs1_tensor, 2)

    def test_custom_func(self):
        def custom_avg(x):
            return sum(x) / len(x)

        ensembled = Ensemble.aggregate(self.obs1_tensor, custom_avg)

        self.assertIsInstance(ensembled, t.Tensor)

        expected = 2 * t.ones([1, 1, 2, 2])
        self.assertTrue(t.equal(ensembled, expected))

        # 2 observations
        ensembled = Ensemble.aggregate((self.obs1_tensor, self.obs2_tensor), custom_avg)
        self.assertIsInstance(ensembled, t.Tensor)

        expected = t.stack([2 * t.ones([1, 2, 2]), 1.5 * t.ones([1, 2, 2])])
        self.assertTrue(t.equal(ensembled, expected))

        def custom_func(x):
            return (3 * x[0] + x[1]) / 6

        ensembled = Ensemble.aggregate(self.obs1_tensor, custom_func)
        expected = t.ones([1, 1, 2, 2])
        self.assertTrue(t.equal(ensembled, expected))

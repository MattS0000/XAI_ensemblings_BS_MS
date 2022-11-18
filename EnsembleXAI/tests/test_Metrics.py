from unittest import TestCase
from EnsembleXAI import Metrics
import torch


class TestHelperFuncs(TestCase):
    images = torch.Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]).repeat(2, 3, 1, 1)
    binary_plus_2d = torch.Tensor([[0, 1, 0],
                                   [1, 1, 1],
                                   [0, 1, 0]])
    binary_cross_2d = torch.Tensor([[1, 0, 1],
                                    [0, 1, 0],
                                    [1, 0, 1]])

    def test_replace_masks(self):
        masks = torch.BoolTensor([[[False, True, False]]]).repeat(2, 3, 1)
        val = 0.1
        replace_masks = Metrics.replace_masks(self.images, masks, value=val)
        self.assertTrue(torch.all(replace_masks[:, :, :, 1] == val))
        self.assertTrue(torch.all(replace_masks[:, :, :, [0, 2]] == self.images[:, :, :, [0, 2]]))

    def test_tensor_to_list_depth1(self):
        list_depth_1 = Metrics.tensor_to_list_tensors(self.images, 1)
        self.assertIsInstance(list_depth_1, list)
        self.assertTrue(len(list_depth_1) == self.images.shape[0])
        self.assertTrue(all([list(list_element.shape) == [3, 3, 3] for list_element in list_depth_1]))
        self.assertTrue(all([isinstance(list_element, torch.Tensor) for list_element in list_depth_1]))

    def test_tensor_to_list_depth2(self):
        list_depth_2 = Metrics.tensor_to_list_tensors(self.images, 2)
        self.assertIsInstance(list_depth_2, list)
        expected_len = self.images.shape[0] * self.images.shape[1]
        self.assertTrue(len(list_depth_2) == expected_len)
        self.assertTrue(all([list(list_element.shape) == [3, 3] for list_element in list_depth_2]))
        self.assertTrue(all([isinstance(list_element, torch.Tensor) for list_element in list_depth_2]))

    def test_matrix_norm_2_basic(self):
        value = Metrics._matrix_norm_2(torch.ones(3, 3), torch.zeros(3, 3))
        self.assertEqual(value, 3)

    def test_matrix_norm_2_tensor(self):
        values = Metrics._matrix_norm_2(torch.ones(3, 3, 3), torch.zeros(3, 3, 3))
        self.assertTrue(torch.all(values == 3))

    def test_matrix_norm_2_deep(self):
        value = Metrics._matrix_norm_2(2 * torch.ones(4, 4, 4), torch.zeros(4, 4, 4), sum_dim=-1)
        self.assertEqual(value, 16)

    def test_intersection(self):
        intersect = Metrics._intersection_mask(self.binary_plus_2d, self.binary_cross_2d)
        self.assertTrue(torch.any(intersect))
        self.assertFalse(torch.all(intersect))
        self.assertTrue(intersect[1, 1])
        intersect[1, 1] = False
        self.assertFalse(torch.any(intersect))
        self.assertFalse(torch.all(intersect))

    def test_intersection_threshold(self):
        plus_threshold_2d = torch.Tensor([[0, 0.2, 0],
                                          [0.5, 0.7, 0.5],
                                          [0, 0.2, 0]])
        blus_threshold_2d = torch.Tensor([[0, 1, 0],
                                          [1, 1, 1],
                                          [0, 1, 0]])
        intersect = Metrics._intersection_mask(plus_threshold_2d, blus_threshold_2d, threshold1=0.3)
        self.assertTrue(torch.all(intersect[1]))
        self.assertFalse(torch.any(intersect[0, 2]))
        reverse_call = Metrics._intersection_mask(blus_threshold_2d, plus_threshold_2d, threshold2=0.3)
        self.assertTrue(torch.all(intersect == reverse_call))

    def test_union(self):
        union = Metrics._union_mask(self.binary_plus_2d, self.binary_cross_2d)
        self.assertTrue(torch.all(union))

    def test_union_threshold(self):
        union_threshold_a = torch.Tensor([[1, 0.4, 1],
                                          [1, 0.4, 1],
                                          [0, 0, 0]])
        union_threshold_b = torch.Tensor([[0, 0, 0],
                                          [0, 0, 0],
                                          [0.6, 0.6, 0.6]])
        union_all = Metrics._union_mask(union_threshold_a, union_threshold_b)
        self.assertTrue(torch.all(union_all))

        union_05 = Metrics._union_mask(union_threshold_a, union_threshold_b, threshold1=0.5)
        correct_union_05 = torch.BoolTensor([[True, False,  True],
                                         [True, False,  True],
                                         [True,  True,  True]])
        self.assertTrue(torch.all(union_05 == correct_union_05))

        union_07 = Metrics._union_mask(union_threshold_a, union_threshold_b, threshold1=0.5, threshold2=0.7)
        correct_union_07 = torch.BoolTensor([[True, False,  True],
                                         [True, False,  True],
                                         [False, False, False]])
        self.assertTrue(torch.all(union_07 == correct_union_07))


class TestMetrics(TestCase):
    pass

from unittest import TestCase
from EnsembleXAI import Ensemble


class TestAggregate(TestCase):
    def test_aggregate(self):
        Ensemble.aggregate()
        self.fail()

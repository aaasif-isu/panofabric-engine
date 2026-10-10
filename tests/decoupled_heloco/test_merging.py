"""Weighting equivalence, heterogeneous batches, RDA and correction ordering."""

import copy
import unittest

import torch

from panoengine.decentralized.decoupled_heloco.config import ConfigError, DecoupledConfig, HeLoCoConfig
from panoengine.decentralized.decoupled_heloco.learner import DecoupledLearner
from panoengine.decentralized.decoupled_heloco.merging import contribution_weights, merge_gradients
from panoengine.decentralized.decoupled_heloco.syncer import DecoupledSyncer


class MergeTests(unittest.TestCase):
    def test_equal_tokens_per_step_reproduces_previous_weights(self):
        self.assertEqual(contribution_weights([16, 40], [2, 5]), contribution_weights([16, 40], [2, 5], "tokens"))

    def test_heterogeneous_tokens_per_step_use_paper_formula(self):
        actual = contribution_weights([16, 80], [2, 5])
        self.assertAlmostEqual(actual[0], 1 / 11)
        self.assertAlmostEqual(actual[1], 10 / 11)
        self.assertNotEqual(actual, contribution_weights([16, 80], [2, 5], "tokens"))

    def test_rda_preserves_weighted_radius_and_handles_zero_or_opposing_directions(self):
        gradients = [torch.tensor([2., 0.]), torch.tensor([0., 4.])]
        actual = merge_gradients(gradients, [.25, .75], "rda")
        torch.testing.assert_close(actual.norm(), torch.tensor(3.5))
        direction = torch.tensor([.25, .75])
        torch.testing.assert_close(actual, 3.5 * direction / direction.norm())
        torch.testing.assert_close(merge_gradients([gradients[0], -gradients[0]], [.5, .5], "rda"), torch.zeros(2))
        torch.testing.assert_close(merge_gradients([torch.zeros(2), torch.zeros(2)], [.5, .5], "rda"), torch.zeros(2))
        torch.testing.assert_close(merge_gradients([gradients[0]], [1.], "rda"), gradients[0])

    def test_paper_weights_and_embedding_merge_are_used_by_syncer(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(1, 2)
                self.other = torch.nn.Parameter(torch.ones(2))
        model = Model()
        models = [copy.deepcopy(model) for _ in range(2)]
        learners = [DecoupledLearner(m, 1, learner_id=i) for i, m in enumerate(models)]
        syncer = DecoupledSyncer(model, learners, 1, min_quorum=2, overlap_steps=1, outer_method="sgd", merge="paper_rda")
        initial = syncer.optimizer.model_snapshot()
        for i, delta in enumerate((torch.tensor([1., 0.]), torch.tensor([0., 1.]))):
            with learners[i].training_step(tokens=8 * (i + 1)), torch.no_grad():
                for p in models[i].parameters():
                    p.sub_(delta.reshape_as(p))
        syncer.begin_sync()
        for learner in learners:
            learner.boundary()
        result = syncer.poll()
        self.assertAlmostEqual(result.weights[0], .2)
        mean = torch.tensor([.2, .8])
        final = syncer.optimizer.model_snapshot()
        torch.testing.assert_close(initial["embedding.weight"] - final["embedding.weight"], mean.reshape(1, 2))
        torch.testing.assert_close(initial["other"] - final["other"], mean / mean.norm())

    def test_correction_order_is_explicit_and_does_not_double_scale_contributions(self):
        outcomes = {}
        for enabled in (True, False):
            for order in ("merge_then_correct", "correct_then_merge"):
                model = torch.nn.Linear(2, 1, bias=False)
                model.weight.data.fill_(1.)
                models = [copy.deepcopy(model) for _ in range(2)]
                learners = [DecoupledLearner(m, 1, learner_id=i) for i, m in enumerate(models)]
                config = HeLoCoConfig(correction_order=order, correction_enabled=enabled, rho=2.)
                syncer = DecoupledSyncer(model, learners, 1, min_quorum=2, overlap_steps=1, heloco=config)
                for deltas in (([1., 0.], [1., 0.]), ([-1., 0.], [0., 1.])):
                    for i, delta in enumerate(deltas):
                        with learners[i].training_step(tokens=8), torch.no_grad():
                            models[i].weight.sub_(torch.tensor([delta]))
                    syncer.begin_sync()
                    for learner in learners:
                        learner.boundary()
                    syncer.poll()
                    for learner in learners:
                        learner.boundary()
                outcomes[enabled, order] = syncer.optimizer.model_snapshot()["weight"]
        # Without directional correction, rho must be applied exactly once in
        # either path. With correction, the nonlinear ordering changes results.
        torch.testing.assert_close(outcomes[False, "merge_then_correct"], outcomes[False, "correct_then_merge"], rtol=0, atol=0)
        self.assertFalse(torch.allclose(outcomes[True, "merge_then_correct"], outcomes[True, "correct_then_merge"]))

    def test_invalid_options_fail_before_training(self):
        for change in ({"weighting": "bad"}, {"merge": "bad"}, {"adaptive_grace": "true"}, {"timing_ema_alpha": 0}, {"heloco": {"correction_order": "bad"}}):
            with self.subTest(change=change), self.assertRaises(ConfigError):
                DecoupledConfig.from_mapping(change)


if __name__ == "__main__":
    unittest.main()

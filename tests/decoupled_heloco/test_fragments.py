"""Fragment coverage and canonical layout checks with CPU PyTorch tensors."""

import unittest

import torch

from panoengine.decentralized.decoupled_heloco.fragment_manager import FragmentManager


class FragmentManagerTests(unittest.TestCase):
    def test_order_and_known_contiguous_partition(self):
        parameters = {f"p{i}": torch.zeros(n) for i, n in enumerate([12, 6, 2, 2, 1])}
        manager = FragmentManager(parameters.items(), 3)
        self.assertEqual([f.parameter_names for f in manager.fragments], [("p0",), ("p1",), ("p2", "p3", "p4")])
        self.assertEqual([f.numel for f in manager.fragments], [12, 6, 5])
        self.assertEqual(manager.total_numel, 23)
        self.assertEqual(dict(manager.parameter_to_fragment), {"p0": 0, "p1": 1, "p2": 2, "p3": 2, "p4": 2})

    def test_every_tensor_covered_once_including_uneven_large_tensors(self):
        for sizes in ([1, 1, 1000, 1, 1], [1000, 1, 1, 1], [0, 0, 0, 0]):
            parameters = {f"p{i}": torch.zeros(n) for i, n in enumerate(sizes)}
            for count in range(1, len(sizes) + 1):
                with self.subTest(sizes=sizes, count=count):
                    manager = FragmentManager(parameters.items(), count)
                    names = [name for fragment in manager.fragments for name in fragment.parameter_names]
                    self.assertEqual(names, list(parameters))
                    self.assertEqual(len(set(names)), len(parameters))
                    self.assertEqual(len(manager.fragments), count)
                    self.assertTrue(all(f.parameter_names for f in manager.fragments))
                    self.assertEqual(manager.total_numel, sum(sizes))

    def test_signature_is_repeatable_but_checks_shape_order_and_fragment_count(self):
        parameters = {"a": torch.zeros(2, 3), "b": torch.zeros(2), "c": torch.zeros(1)}
        manager = FragmentManager(parameters.items(), 2)
        bf16 = {name: tensor.to(torch.bfloat16) for name, tensor in parameters.items()}
        self.assertEqual(manager.layout_signature, FragmentManager(bf16.items(), 2).layout_signature)
        reversed_names = dict(reversed(list(parameters.items())))
        self.assertNotEqual(manager.layout_signature, FragmentManager(reversed_names.items(), 2).layout_signature)
        self.assertNotEqual(manager.layout_signature, FragmentManager(parameters.items(), 1).layout_signature)
        reshaped = {**parameters, "a": parameters["a"].reshape(3, 2)}
        self.assertNotEqual(manager.layout_signature, FragmentManager(reshaped.items(), 2).layout_signature)
        with self.assertRaises(ValueError):
            manager.validate_model(reshaped)

    def test_invalid_layouts_are_rejected(self):
        parameter = torch.zeros(2)
        for count in (0, -1, 1.5, True, 3):
            with self.subTest(count=count), self.assertRaises(ValueError):
                FragmentManager([("a", parameter), ("b", torch.zeros(1))], count)
        with self.assertRaises(ValueError):
            FragmentManager([], 1)
        with self.assertRaises(ValueError):
            FragmentManager([("a", parameter), ("a", torch.zeros(1))], 1)
        with self.assertRaises(ValueError):
            FragmentManager([("a", parameter), ("alias", parameter)], 1)

    def test_model_factory_deduplicates_tied_parameters_and_includes_frozen_ones(self):
        model = torch.nn.Module()
        model.first = torch.nn.Linear(2, 2)
        model.second = model.first
        model.first.bias.requires_grad_(False)
        manager = FragmentManager.from_model(model, 2)
        self.assertEqual([s.name for s in manager.parameter_specs], ["first.weight", "first.bias"])
        self.assertEqual(manager.total_numel, 6)

    def test_meta_layout_needs_no_parameter_storage(self):
        model = torch.nn.Linear(4, 3, device="meta")
        manager = FragmentManager.from_model(model, 2)
        self.assertEqual(manager.total_numel, 15)
        self.assertEqual(manager.fragment(0).parameter_names, ("weight",))
        for fragment_id in (-1, 2, True):
            with self.assertRaises(ValueError):
                manager.fragment(fragment_id)


if __name__ == "__main__":
    unittest.main()

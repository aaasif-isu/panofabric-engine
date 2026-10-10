"""Token IDs must fit the model; a tokenizer need not fill every model slot."""

from types import SimpleNamespace
import unittest

from panoengine.decentralized.decoupled_heloco.config import ConfigError
from panoengine.decentralized.decoupled_heloco.gpu_recipe import validate_tokenizer


class TokenizerTests(unittest.TestCase):
    def tokenizer(self, count=2020):
        vocab = {f"token_{i}": i for i in range(count)}
        return SimpleNamespace(get_vocab=lambda: vocab, encode=lambda text, **kwargs: [1999, 0, 2000], bos_id=1999, eos_id=2000)

    def test_smaller_debug_vocabulary_and_added_special_tokens_fit(self):
        tokenizer = self.tokenizer()
        self.assertEqual(validate_tokenizer(tokenizer, 2048), (2020, 2019))
        tokenizer.get_vocab()["[PAD]"] = 2020
        self.assertEqual(validate_tokenizer(tokenizer, 2048), (2021, 2020))

    def test_exact_capacity_including_the_last_slot_fits(self):
        self.assertEqual(validate_tokenizer(self.tokenizer(2048), 2048), (2048, 2047))

    def test_larger_real_model_tokenizer_is_rejected_with_actual_range(self):
        with self.assertRaisesRegex(ConfigError, "vocabulary=128256, max_token_id=128255"):
            validate_tokenizer(self.tokenizer(128256), 2048)

    def test_small_vocabulary_with_a_sparse_high_id_is_rejected(self):
        tokenizer = self.tokenizer()
        tokenizer.get_vocab = lambda: {"a": 0, "b": 2048}
        with self.assertRaisesRegex(ConfigError, "max_token_id=2048"):
            validate_tokenizer(tokenizer, 2048)

    def test_special_and_emitted_ids_are_checked_separately(self):
        tokenizer = self.tokenizer()
        tokenizer.bos_id = 2048
        with self.assertRaisesRegex(ConfigError, "bos_id=2048"):
            validate_tokenizer(tokenizer, 2048)
        tokenizer.bos_id = 1999
        tokenizer.encode = lambda *args, **kwargs: [0, 2048]
        with self.assertRaisesRegex(ConfigError, "emitted invalid IDs"):
            validate_tokenizer(tokenizer, 2048)

    def test_empty_negative_and_noninteger_vocabularies_are_rejected(self):
        tokenizer = self.tokenizer()
        for vocab in ({}, {"a": -1}, {"a": True}, {"a": "1"}):
            with self.subTest(vocab=vocab), self.assertRaises(ConfigError):
                tokenizer.get_vocab = lambda: vocab
                validate_tokenizer(tokenizer, 2048)


if __name__ == "__main__":
    unittest.main()

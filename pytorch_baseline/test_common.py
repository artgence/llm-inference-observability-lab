#!/usr/bin/env python3
"""Dependency-free tests for shared PyTorch-baseline helpers."""

from __future__ import annotations

import unittest

from common import normalize_token_ids, render_chat_ids


class TensorLike:
    def __init__(self, value: object) -> None:
        self.value = value

    def tolist(self) -> object:
        return self.value


class ChatTokenizer:
    chat_template = "test-template"

    def __init__(self, result: object) -> None:
        self.result = result

    def apply_chat_template(self, *_: object, **__: object) -> object:
        return self.result


class TokenIdNormalizationTests(unittest.TestCase):
    def test_flat_list(self) -> None:
        self.assertEqual(normalize_token_ids([1, 2, 3]), [1, 2, 3])

    def test_batched_tensor(self) -> None:
        self.assertEqual(normalize_token_ids(TensorLike([[1, 2, 3]])), [1, 2, 3])

    def test_mapping_returned_by_chat_template(self) -> None:
        tokenizer = ChatTokenizer({"input_ids": TensorLike([[4, 5, 6]])})
        self.assertEqual(render_chat_ids(tokenizer, "hello"), [4, 5, 6])

    def test_multiple_prompts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one prompt"):
            normalize_token_ids([[1, 2], [3, 4]])

    def test_mapping_without_input_ids_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "input_ids"):
            normalize_token_ids({"attention_mask": [1, 1]})


if __name__ == "__main__":
    unittest.main()

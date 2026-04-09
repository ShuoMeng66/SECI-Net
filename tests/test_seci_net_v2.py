from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from api_server import ModelRuntime
from core.model import HybridTextClassifier, build_model_from_checkpoint, save_checkpoint_bundle


class SECINetV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = HybridTextClassifier(
            vocab_size=32,
            num_classes=2,
            embed_dim=16,
            transformer_layers=1,
            num_heads=4,
            ffn_hidden_dim=24,
            lstm_hidden_dim=8,
            lstm_layers=1,
            dropout=0.0,
            max_len=32,
            block_size=4,
            topk_global_blocks=2,
            use_temporal_encoding=True,
            positive_class_index=1,
        )

    def test_return_dict_outputs_have_expected_shapes(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.long)
        lengths = attention_mask.sum(dim=1)
        time_values = torch.tensor([1.0, 2.0], dtype=torch.float)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lengths=lengths,
            time_values=time_values,
            return_dict=True,
        )

        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertEqual(tuple(output.features.shape), (2, 24))
        self.assertEqual(tuple(output.token_features.shape), (2, 5, 16))
        self.assertEqual(tuple(output.evidence_indices.shape), (2, 2))
        self.assertEqual(tuple(output.evidence_scores.shape), (2, 2))
        self.assertEqual(tuple(output.recoverability.shape), (2,))

    def test_router_masks_out_invalid_blocks(self) -> None:
        input_ids = torch.tensor([[1, 2, 0, 0, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.long)
        lengths = attention_mask.sum(dim=1)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lengths=lengths,
            return_dict=True,
        )

        invalid_mask = ~output.block_mask
        self.assertTrue(torch.all(output.block_scores[invalid_mask] < -1e3))
        self.assertAlmostEqual(float(output.router_probabilities.sum().item()), 1.0, places=5)

    def test_checkpoint_bundle_reload_and_runtime_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "best_model.pt"
            vocab = {
                "min_freq": 1,
                "max_size": None,
                "specials": ["<pad>", "<unk>"],
                "itos": ["<pad>", "<unk>", "服", "务", "差", "好", "物", "流"],
            }
            labels = {"0": 0, "1": 1}
            save_checkpoint_bundle(
                checkpoint_path,
                model=self.model,
                vocab=vocab,
                label_to_index=labels,
            )

            loaded_model, payload = build_model_from_checkpoint(checkpoint_path, map_location="cpu")
            self.assertEqual(payload["architecture"], "seci-net-v2")
            self.assertEqual(loaded_model.get_model_config()["block_size"], 4)

            runtime = ModelRuntime(checkpoint_path=checkpoint_path, device="cpu")
            analysis = runtime.predict("服务好但是物流慢")
            self.assertIn("recoverability", analysis)
            self.assertIn("probabilities", analysis)
            self.assertTrue(isinstance(analysis["evidence"], list))
            self.assertGreaterEqual(len(analysis["evidence"]), 1)


if __name__ == "__main__":
    unittest.main()

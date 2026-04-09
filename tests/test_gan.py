from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from GAN import (
    CounterfactualReviewDataset,
    CounterfactualReviewGAN,
    CounterfactualReviewRecord,
    GANTrainingConfig,
    ReviewGANTrainer,
    build_aspect_mapping,
    build_gan_vocab,
    build_label_mapping,
    collate_counterfactual_reviews,
)
from main.train import main as train_main


class GANModuleTests(unittest.TestCase):
    def build_records(self) -> list[CounterfactualReviewRecord]:
        return [
            CounterfactualReviewRecord(
                text="service was great",
                label="positive",
                counterfactual_text="service was rude",
                counterfactual_label="negative",
            ),
            CounterfactualReviewRecord(
                text="delivery was slow",
                label="negative",
                counterfactual_text="delivery was fast",
                counterfactual_label="positive",
                source_aspect="logistics",
                target_aspect="logistics",
            ),
        ]

    def build_bundle(self):
        records = self.build_records()
        vocab = build_gan_vocab(records)
        label_to_index = build_label_mapping(records)
        aspect_to_index = build_aspect_mapping(records)
        dataset = CounterfactualReviewDataset(
            records=records,
            vocab=vocab,
            label_to_index=label_to_index,
            aspect_to_index=aspect_to_index,
            max_source_len=16,
            max_target_len=16,
        )
        batch = collate_counterfactual_reviews([dataset[0], dataset[1]], pad_idx=vocab.pad_idx)
        config = GANTrainingConfig(
            embed_dim=16,
            hidden_dim=12,
            noise_dim=8,
            label_embed_dim=8,
            aspect_embed_dim=8,
            time_embed_dim=4,
            max_source_len=16,
            max_target_len=16,
            batch_size=2,
        )
        return records, vocab, label_to_index, aspect_to_index, dataset, batch, config

    def test_dataset_and_collate_fallback_for_missing_aspect_and_time(self) -> None:
        records, vocab, _, aspect_to_index, dataset, batch, _ = self.build_bundle()

        self.assertEqual(dataset[0]["source_aspect"], aspect_to_index["generic"])
        self.assertEqual(dataset[0]["target_aspect"], aspect_to_index["generic"])
        self.assertEqual(float(batch["time_values"][0].item()), 0.0)
        self.assertEqual(float(batch["counterfactual_time_values"][0].item()), 0.0)
        self.assertEqual(batch["source_input_ids"].shape[0], len(records))

    def test_generator_and_discriminator_forward_shapes(self) -> None:
        _, vocab, label_to_index, aspect_to_index, _, batch, config = self.build_bundle()
        gan = CounterfactualReviewGAN(
            vocab_size=len(vocab),
            num_labels=len(label_to_index),
            num_aspects=len(aspect_to_index),
            config=config,
            pad_idx=vocab.pad_idx,
        )

        generator_output = gan.generator(
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            source_lengths=batch["source_lengths"],
            decoder_input_ids=batch["decoder_input_ids"],
            target_labels=batch["target_labels"],
            target_aspects=batch["target_aspects"],
            time_values=batch["time_values"],
            counterfactual_time_values=batch["counterfactual_time_values"],
        )
        discriminator_output = gan.discriminator(
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            candidate_input_ids=batch["target_input_ids"],
            candidate_attention_mask=batch["target_attention_mask"],
            target_labels=batch["target_labels"],
            target_aspects=batch["target_aspects"],
        )

        self.assertEqual(generator_output["token_logits"].shape[0], 2)
        self.assertEqual(generator_output["token_logits"].shape[1], batch["target_input_ids"].shape[1])
        self.assertEqual(generator_output["token_logits"].shape[2], len(vocab))
        self.assertEqual(generator_output["edit_probabilities"].shape, batch["source_attention_mask"].shape)
        self.assertEqual(tuple(discriminator_output["real_fake_logits"].shape), (2,))
        self.assertEqual(tuple(discriminator_output["label_logits"].shape), (2, len(label_to_index)))

    def test_trainer_train_epoch_and_export(self) -> None:
        torch.manual_seed(7)
        records, vocab, label_to_index, aspect_to_index, dataset, _, config = self.build_bundle()
        gan = CounterfactualReviewGAN(
            vocab_size=len(vocab),
            num_labels=len(label_to_index),
            num_aspects=len(aspect_to_index),
            config=config,
            pad_idx=vocab.pad_idx,
        )
        trainer = ReviewGANTrainer(gan=gan, vocab=vocab, config=config, device="cpu")
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=lambda items: collate_counterfactual_reviews(items, vocab.pad_idx),
        )

        metrics = trainer.train_epoch(loader)
        exported = trainer.export_stage2_pairs(records)

        self.assertIn("generator_loss", metrics)
        self.assertGreaterEqual(metrics["generator_reconstruction_loss"], 0.0)
        self.assertEqual(len(exported), len(records))
        self.assertTrue(all(row["counterfactual_text"] for row in exported))
        self.assertTrue(all(row["counterfactual_label"] in {"positive", "negative"} for row in exported))


class TrainGANIntegrationTests(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["text", "label", "counterfactual_text", "counterfactual_label"],
            )
            writer.writeheader()
            writer.writerows(rows)

    def train_args(self, train_path: Path, save_dir: Path, *extra: str) -> list[str]:
        return [
            "--train_path",
            str(train_path),
            "--save_dir",
            str(save_dir),
            "--epochs",
            "1",
            "--batch_size",
            "2",
            "--embed_dim",
            "8",
            "--transformer_layers",
            "1",
            "--num_heads",
            "2",
            "--ffn_hidden_dim",
            "16",
            "--lstm_hidden_dim",
            "4",
            "--lstm_layers",
            "1",
            "--dropout",
            "0.0",
            "--max_len",
            "16",
            "--block_size",
            "4",
            "--local_window_size",
            "2",
            "--topk_global_blocks",
            "1",
            "--num_workers",
            "0",
            "--log_every_steps",
            "0",
            "--no_amp",
            "--gan_epochs",
            "1",
            "--gan_batch_size",
            "2",
            "--gan_max_source_len",
            "16",
            "--gan_max_target_len",
            "16",
            *extra,
        ]

    def test_train_smoke_without_gan_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.csv"
            save_dir = root / "run"
            self.write_csv(
                train_path,
                [
                    {"text": "service was great", "label": "positive", "counterfactual_text": "", "counterfactual_label": ""},
                    {"text": "delivery was slow", "label": "negative", "counterfactual_text": "", "counterfactual_label": ""},
                ],
            )

            train_main(self.train_args(train_path, save_dir))

            self.assertTrue((save_dir / "best_model.pt").exists())
            self.assertFalse((save_dir / "gan").exists())

    def test_train_smoke_with_gan_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.csv"
            save_dir = root / "run"
            self.write_csv(
                train_path,
                [
                    {
                        "text": "service was great",
                        "label": "positive",
                        "counterfactual_text": "service was rude",
                        "counterfactual_label": "negative",
                    },
                    {
                        "text": "delivery was slow",
                        "label": "negative",
                        "counterfactual_text": "delivery was fast",
                        "counterfactual_label": "positive",
                    },
                    {
                        "text": "price was fair",
                        "label": "positive",
                        "counterfactual_text": "price was expensive",
                        "counterfactual_label": "negative",
                    },
                    {
                        "text": "packaging arrived damaged",
                        "label": "negative",
                        "counterfactual_text": "",
                        "counterfactual_label": "",
                    },
                ],
            )

            train_main(
                self.train_args(
                    train_path,
                    save_dir,
                    "--enable_gan_augmentation",
                )
            )

            self.assertTrue((save_dir / "best_model.pt").exists())
            self.assertTrue((save_dir / "gan" / "generated_counterfactuals.csv").exists())
            self.assertTrue((save_dir / "gan" / "metrics.json").exists())

    def test_train_smoke_with_no_paired_records_warns_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.csv"
            save_dir = root / "run"
            self.write_csv(
                train_path,
                [
                    {"text": "app kept crashing", "label": "negative", "counterfactual_text": "", "counterfactual_label": ""},
                    {"text": "staff solved it quickly", "label": "positive", "counterfactual_text": "", "counterfactual_label": ""},
                ],
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                train_main(
                    self.train_args(
                        train_path,
                        save_dir,
                        "--enable_gan_augmentation",
                    )
                )

            self.assertIn("Skipping GAN augmentation", stdout.getvalue())
            self.assertTrue((save_dir / "best_model.pt").exists())
            self.assertTrue((save_dir / "gan" / "generated_counterfactuals.csv").exists())


if __name__ == "__main__":
    unittest.main()

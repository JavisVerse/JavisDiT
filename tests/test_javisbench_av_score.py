import importlib
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import Dataset

from javisdit.evaluation import javisbench_av_score as av_score


class _IdentityImageBind(torch.nn.Module):
    def forward(self, inputs):
        return inputs


class _NonFiniteImageBind(torch.nn.Module):
    def forward(self, inputs):
        return {key: torch.full_like(value, float("nan")) for key, value in inputs.items()}


class _FakeAVScoreDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.samples = [
            (
                {
                    "vision": torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]),
                    "audio": torch.tensor([[1.0, 0.0]]),
                },
                {"audio": torch.tensor([[1.0, 0.0], [1.0, 0.0]])},
                torch.tensor([[0, 1], [2, 3]]),
            ),
            (
                {
                    "vision": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
                    "audio": torch.tensor([[1.0, 0.0]]),
                },
                {"audio": torch.tensor([[1.0, 0.0]])},
                torch.tensor([[0, 1]]),
            ),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class EvaluateAVScoreTest(unittest.TestCase):
    def test_import_does_not_mutate_grad_mode_or_logging(self):
        grad_enabled = torch.is_grad_enabled()
        warning_function = logging.warning

        importlib.reload(av_score)

        self.assertEqual(torch.is_grad_enabled(), grad_enabled)
        self.assertIs(logging.warning, warning_function)

    def test_returns_standalone_avh_and_javis_means(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            videos = [root / "0.mp4", root / "1.mp4"]
            audios = [root / "0.wav", root / "1.wav"]
            checkpoint = root / "imagebind_huge.pth"
            for path in [*videos, *audios, checkpoint]:
                path.touch()

            with (
                mock.patch.object(av_score, "_AVScoreDataset", _FakeAVScoreDataset),
                mock.patch.object(
                    av_score,
                    "_load_imagebind_model",
                    return_value=_IdentityImageBind(),
                ),
            ):
                result = av_score.evaluate_av_score(
                    videos,
                    audios,
                    imagebind_checkpoint=checkpoint,
                    device="cpu",
                    topk_min=0.4,
                )

        self.assertEqual(set(result), {"avh_score", "javis_score"})
        self.assertAlmostEqual(result["avh_score"], 0.5)
        self.assertAlmostEqual(result["javis_score"], 0.25)
        self.assertTrue(all(isinstance(value, float) for value in result.values()))

    def test_rejects_unpaired_inputs_before_loading_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "0.mp4"
            audio = root / "0.wav"
            checkpoint = root / "imagebind_huge.pth"
            for path in (video, audio, checkpoint):
                path.touch()

            with mock.patch.object(av_score, "_load_imagebind_model") as loader:
                with self.assertRaisesRegex(ValueError, "equal length"):
                    av_score.evaluate_av_score(
                        [video, video],
                        [audio],
                        imagebind_checkpoint=checkpoint,
                        device="cpu",
                    )
                loader.assert_not_called()

    def test_missing_checkpoint_does_not_download_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "0.mp4"
            audio = root / "0.wav"
            video.touch()
            audio.touch()

            with mock.patch.object(torch.hub, "download_url_to_file") as download:
                with self.assertRaisesRegex(FileNotFoundError, "allow_download=True"):
                    av_score.evaluate_av_score(
                        [video],
                        [audio],
                        imagebind_checkpoint=root / "cache",
                        device="cpu",
                    )
                download.assert_not_called()

    def test_rejects_non_finite_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "0.mp4"
            audio = root / "0.wav"
            checkpoint = root / "imagebind_huge.pth"
            for path in (video, audio, checkpoint):
                path.touch()

            with (
                mock.patch.object(av_score, "_AVScoreDataset", _FakeAVScoreDataset),
                mock.patch.object(
                    av_score,
                    "_load_imagebind_model",
                    return_value=_NonFiniteImageBind(),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    av_score.evaluate_av_score(
                        [video],
                        [audio],
                        imagebind_checkpoint=checkpoint,
                        device="cpu",
                        topk_min=0.5,
                    )

    def test_cli_reads_json_manifest_and_writes_two_scores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            output = root / "scores.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "video_path": "",
                            "pred_video_path": "sample.mp4",
                            "audio_path": "",
                            "pred_audio_path": "sample.wav",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            expected = {"avh_score": 0.1, "javis_score": 0.2}

            with (
                mock.patch.object(av_score, "evaluate_av_score", return_value=expected) as evaluate,
                mock.patch.object(av_score.os, "replace", wraps=av_score.os.replace) as replace,
            ):
                status = av_score.main(
                    [
                        "--input-file",
                        str(manifest),
                        "--output-file",
                        str(output),
                        "--imagebind-checkpoint",
                        str(root / "imagebind_huge.pth"),
                        "--device",
                        "cpu",
                        "--window-size-s",
                        "2.0",
                        "--window-overlap-s",
                        "1.5",
                        "--topk-min",
                        "0.4",
                        "--num-workers",
                        "0",
                    ]
                )

            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), expected)
            evaluate.assert_called_once()
            replace.assert_called_once()


if __name__ == "__main__":
    unittest.main()

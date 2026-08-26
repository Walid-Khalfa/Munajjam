"""
Unit tests for ``scripts/export_fastconformer_onnx.py``.

No NeMo / torch / onnxruntime are required: the heavy export path is mocked
and the tokenizer extraction is tested against tiny synthetic ``.nemo``
archives.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts import export_fastconformer_onnx as exporter

STEM = "stt_ar_fastconformer_hybrid_large_pc_v1.0"


# --------------------------------------------------------------------------- #
# CLI parsing
# --------------------------------------------------------------------------- #
def test_parse_args_requires_checkpoint() -> None:
    with pytest.raises(SystemExit):
        exporter.parse_args([])


def test_parse_args_defaults(tmp_path: Path) -> None:
    args = exporter.parse_args([str(tmp_path / "model.nemo")])
    assert args.checkpoint == tmp_path / "model.nemo"
    assert args.output_dir == exporter.DEFAULT_OUTPUT_DIR
    assert args.opset == 18
    assert args.force is False
    assert args.no_validate is False


def test_parse_args_output_dir_and_flags(tmp_path: Path) -> None:
    out = tmp_path / "models"
    args = exporter.parse_args(
        ["model.nemo", "--output-dir", str(out), "--force", "--no-validate", "--opset", "17"]
    )
    assert args.output_dir == out
    assert args.force is True
    assert args.no_validate is True
    assert args.opset == 17


# --------------------------------------------------------------------------- #
# Tokenizer / vocabulary extraction from a .nemo archive
# --------------------------------------------------------------------------- #
def _make_nemo_archive(
    tmp_path: Path,
    *,
    name: str | None = None,
    include_tokenizer: bool = True,
    include_vocab: bool = True,
    config_reference: str = "abc123_tokenizer.model",
) -> Path:
    path = tmp_path / (name or f"{STEM}.nemo")
    with tarfile.open(path, "w:gz") as tar:
        config = f"tokenizer:\n  model: {config_reference}\n  type: sentencepiece\n"
        _add_bytes(tar, "model_config.yaml", config.encode())
        if include_tokenizer:
            _add_bytes(tar, config_reference, b"fake-sentencepiece-model")
        if include_vocab:
            _add_bytes(tar, "vocab.txt", b"tok1\ntok2\n")
    return path


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_extract_assets_writes_tokenizer_and_vocab(tmp_path: Path) -> None:
    nemo = _make_nemo_archive(tmp_path)
    out = tmp_path / "out"

    assets = exporter.extract_assets(nemo, out)

    assert assets["tokenizer_model"] == out / exporter.TOKENIZER_FILENAME
    assert assets["tokenizer_model"].read_bytes() == b"fake-sentencepiece-model"
    assert assets["vocabulary"] == out / exporter.VOCAB_FILENAME
    assert assets["vocabulary"].read_text() == "tok1\ntok2\n"


def test_extract_assets_uses_config_referenced_tokenizer(tmp_path: Path) -> None:
    """The tokenizer referenced by model_config.yaml wins over other *.model files."""
    nemo = tmp_path / f"{STEM}.nemo"
    with tarfile.open(nemo, "w:gz") as tar:
        _add_bytes(tar, "model_config.yaml", b"tokenizer:\n  model: real_tokenizer.model\n")
        _add_bytes(tar, "real_tokenizer.model", b"real")
        _add_bytes(tar, "stale_tokenizer.model", b"stale")
    out = tmp_path / "out"

    assets = exporter.extract_assets(nemo, out)
    assert assets["tokenizer_model"].read_bytes() == b"real"


def test_extract_assets_without_config_falls_back_to_glob(tmp_path: Path) -> None:
    nemo = tmp_path / f"{STEM}.nemo"
    with tarfile.open(nemo, "w:gz") as tar:
        _add_bytes(tar, "model_config.yaml", b"tokenizer:\n  type: sentencepiece\n")
        _add_bytes(tar, "deadbeef_tokenizer.model", b"fallback")
    out = tmp_path / "out"

    assets = exporter.extract_assets(nemo, out)
    assert assets["tokenizer_model"].read_bytes() == b"fallback"


def test_extract_assets_missing_tokenizer_raises(tmp_path: Path) -> None:
    nemo = _make_nemo_archive(tmp_path, include_tokenizer=False)
    with pytest.raises(SystemExit, match="no SentencePiece tokenizer"):
        exporter.extract_assets(nemo, tmp_path / "out")


def test_extract_assets_invalid_archive_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.nemo"
    bogus.write_bytes(b"not a tar")
    with pytest.raises(SystemExit, match="not a readable .nemo"):
        exporter.extract_assets(bogus, tmp_path / "out")


def test_extract_assets_refuses_overwrite_without_force(tmp_path: Path) -> None:
    nemo = _make_nemo_archive(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / exporter.TOKENIZER_FILENAME).write_bytes(b"existing")

    with pytest.raises(SystemExit, match="already exists"):
        exporter.extract_assets(nemo, out)

    assert (out / exporter.TOKENIZER_FILENAME).read_bytes() == b"existing"


def test_extract_assets_overwrites_with_force(tmp_path: Path) -> None:
    nemo = _make_nemo_archive(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / exporter.TOKENIZER_FILENAME).write_bytes(b"existing")

    exporter.extract_assets(nemo, out, force=True)
    assert (out / exporter.TOKENIZER_FILENAME).read_bytes() == b"fake-sentencepiece-model"


# --------------------------------------------------------------------------- #
# Export-time dependencies
# --------------------------------------------------------------------------- #
def test_import_export_deps_missing_raises_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="nemo_toolkit"):
        exporter._import_export_deps()


def test_export_onnx_graphs_wraps_nemo_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = MagicMock()
    fake_nemo = MagicMock()
    fake_nemo.models.EncDecHybridRNNTCTCBPEModel.restore_from.side_effect = RuntimeError(
        "not a nemo checkpoint"
    )
    monkeypatch.setattr(exporter, "_import_export_deps", lambda: (fake_torch, fake_nemo))

    with pytest.raises(RuntimeError, match="Failed to restore checkpoint"):
        exporter.export_onnx_graphs(Path("x.nemo"), Path("out"))


# --------------------------------------------------------------------------- #
# main() end-to-end (heavy export mocked)
# --------------------------------------------------------------------------- #
def test_main_missing_checkpoint_returns_2(tmp_path: Path) -> None:
    code = exporter.main([str(tmp_path / "missing.nemo"), "--output-dir", str(tmp_path / "out")])
    assert code == 2


def test_main_extracts_assets_and_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nemo = _make_nemo_archive(tmp_path)
    out = tmp_path / "out"

    def fake_export(nemo_path, output_dir, *, opset, force):
        raw = output_dir / f"{nemo_path.stem}{exporter.RAW_AUDIO_ONNX_SUFFIX}"
        raw.write_bytes(b"onnx")
        stock = output_dir / f"{nemo_path.stem}{exporter.STOCK_ONNX_SUFFIX}"
        stock.write_bytes(b"onnx")
        return {"stock": stock, "raw_audio": raw}

    monkeypatch.setattr(exporter, "export_onnx_graphs", fake_export)

    code = exporter.main([str(nemo), "--output-dir", str(out), "--no-validate"])
    assert code == 0

    # Naming contract: deterministic, matches the runtime's expectations.
    assert (out / f"{STEM}{exporter.STOCK_ONNX_SUFFIX}").is_file()
    assert (out / f"{STEM}{exporter.RAW_AUDIO_ONNX_SUFFIX}").is_file()
    assert (out / exporter.TOKENIZER_FILENAME).is_file()
    assert (out / exporter.VOCAB_FILENAME).is_file()


def test_main_refuses_existing_output_without_force(tmp_path: Path) -> None:
    nemo = _make_nemo_archive(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / exporter.TOKENIZER_FILENAME).write_bytes(b"existing")

    with pytest.raises(SystemExit, match="already exists"):
        exporter.main([str(nemo), "--output-dir", str(out), "--no-validate"])
    assert (out / exporter.TOKENIZER_FILENAME).read_bytes() == b"existing"


# --------------------------------------------------------------------------- #
# Post-export validation
# --------------------------------------------------------------------------- #
def test_validate_onnx_checks_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / f"{STEM}_ctc_rawaudio.onnx"
    raw.write_bytes(b"onnx")

    class FakeIO:
        def __init__(self, name, type_):
            self.name = name
            self.type = type_

    class FakeSession:
        def __init__(self):
            self._inputs = [
                FakeIO(exporter.INPUT_SIGNAL_NAME, "tensor(float)"),
                FakeIO(exporter.INPUT_LENGTH_NAME, "tensor(int32)"),
            ]
            self._outputs = [
                FakeIO(exporter.OUTPUT_LOGPROBS_NAME, "tensor(float)"),
                FakeIO(exporter.OUTPUT_LENGTH_NAME, "tensor(int64)"),
            ]

        def get_inputs(self):
            return self._inputs

        def get_outputs(self):
            return self._outputs

        def run(self, output_names, input_feed):
            import numpy as np

            return [np.zeros((1, 2, 1025), dtype=np.float32), np.array([2], dtype=np.int64)]

    fake_ort = MagicMock()
    fake_ort.InferenceSession.return_value = FakeSession()
    import sys

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    # numpy is required by validate_onnx; ensure it resolves.
    exporter.validate_onnx(raw)


def test_validate_onnx_rejects_wrong_input_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / f"{STEM}_ctc_rawaudio.onnx"
    raw.write_bytes(b"onnx")

    fake_ort = MagicMock()
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="audio_signal", type="tensor(float)")]
    fake_ort.InferenceSession.return_value = session
    import sys

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(SystemExit, match="unexpected graph inputs"):
        exporter.validate_onnx(raw)

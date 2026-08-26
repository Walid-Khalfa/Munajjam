#!/usr/bin/env python3
"""
Export the FastConformer CTC ONNX graph and tokenizer assets from a NeMo
checkpoint.

This is the **export-time** companion of
``munajjam/transcription/fastconformer.py`` (``FastConformerInference``). It
turns the reference checkpoint
``nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0`` (a NeMo
``EncDecHybridRNNTCTCBPEModel``, ~424 MB ``.nemo``) into the ONNX artifacts
the runtime expects. See ``docs/fastconformer-onnx-validation.md`` for the
verified ONNX contract and numerical parity results.

Two graphs are produced:

* ``{stem}_ctc.onnx`` — NeMo's *stock* ``model.export()`` after
  ``change_decoding_strategy(decoder_type="ctc")``. Takes log-mel features
  (``audio_signal [B, 80, T_mel]`` + ``length [B]``); **not** consumed by
  ``FastConformerInference``.
* ``{stem}_ctc_rawaudio.onnx`` — the Munajjam *production* graph: a single
  self-contained graph (preprocessor -> encoder -> CTC head) that accepts raw
  16 kHz mono waveforms directly. This is the file the runtime loads.

The SentencePiece tokenizer baked into the ``.nemo`` (and ``vocabulary.txt``
when the archive carries it) is extracted alongside so the runtime has
everything it needs.

CLI contract::

    python scripts/export_fastconformer_onnx.py <checkpoint.nemo> \\
        --output-dir .model_validation/fastconformer

Generated files (for the reference checkpoint, ``stem`` is
``stt_ar_fastconformer_hybrid_large_pc_v1.0``)::

    {output_dir}/{stem}_ctc.onnx                 stock mel-input graph (optional)
    {output_dir}/{stem}_ctc_rawaudio.onnx        production raw-audio graph
    {output_dir}/{stem}_ctc_rawaudio.onnx.data   external weights (large models)
    {output_dir}/tokenizer.model                 SentencePiece model
    {output_dir}/vocabulary.txt                  labels dump (when present)

Existing output files are never overwritten unless ``--force`` is passed.

NeMo/torch are **export-time only** dependencies: they are imported lazily
inside the export path and are never required by the munajjam runtime.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# ONNX contract constants (must match transcription/fastconformer.py)
# --------------------------------------------------------------------------- #
INPUT_SIGNAL_NAME = "input_signal"          # float32 [B, T] raw waveform @16 kHz
INPUT_LENGTH_NAME = "input_signal_length"   # int32   [B] valid sample count
OUTPUT_LOGPROBS_NAME = "logprobs"           # float32 [B, T', V+1] (log-softmax)
OUTPUT_LENGTH_NAME = "encoded_lengths"      # int64   [B] true frame count
SAMPLE_RATE = 16000
DEFAULT_OPSET = 18

# Documented I/O types of the production raw-audio graph (NodeArg.type),
# validated by validate_onnx() before any inference is run.
EXPECTED_IO_TYPES = {
    INPUT_SIGNAL_NAME: "tensor(float)",
    INPUT_LENGTH_NAME: "tensor(int32)",
    OUTPUT_LOGPROBS_NAME: "tensor(float)",
    OUTPUT_LENGTH_NAME: "tensor(int64)",
}

STOCK_ONNX_SUFFIX = "_ctc.onnx"
RAW_AUDIO_ONNX_SUFFIX = "_ctc_rawaudio.onnx"
TOKENIZER_FILENAME = "tokenizer.model"
VOCAB_FILENAME = "vocabulary.txt"

DEFAULT_OUTPUT_DIR = Path(".model_validation/fastconformer")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (importable so unit tests can drive it)."""
    parser = argparse.ArgumentParser(
        prog="export_fastconformer_onnx.py",
        description=(
            "Export the FastConformer CTC ONNX graph and SentencePiece "
            "tokenizer from a NeMo .nemo checkpoint for Munajjam CTC alignment."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the NeMo checkpoint (.nemo), e.g. "
        "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where ONNX graphs and tokenizer assets are written.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help="ONNX opset version for the production raw-audio export.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files (default: refuse).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the post-export onnxruntime sanity check.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Tokenizer / vocabulary extraction (no NeMo needed)
# --------------------------------------------------------------------------- #
def _member_reader(archive: tarfile.TarFile, name: str) -> bytes | None:
    """Read a member's raw bytes, or ``None`` if absent/not a regular file."""
    try:
        member = archive.getmember(name)
    except KeyError:
        return None
    if not member.isfile():
        return None
    f = archive.extractfile(member)
    if f is None:
        return None
    with f:
        return f.read()


def _find_tokenizer_member(archive: tarfile.TarFile) -> str | None:
    """Locate the SentencePiece model inside a .nemo archive.

    Deterministic preference:
    1. the file referenced by ``model_config.yaml``'s ``tokenizer.model``;
    2. any archive member named ``*_tokenizer.model`` / ``*tokenizer.model``.
    """
    names = archive.getnames()
    config = _member_reader(archive, "model_config.yaml")
    if config is not None:
        referenced = _tokenizer_path_from_config(config)
        if referenced and referenced in names:
            return referenced

    candidates = [n for n in names if n.endswith(("_tokenizer.model", "/tokenizer.model"))]
    # Prefer the shortest path (archive root) for determinism.
    return min(candidates, key=len) if candidates else None


def _tokenizer_path_from_config(config_bytes: bytes) -> str | None:
    """Read ``tokenizer.model`` from a NeMo ``model_config.yaml``.

    Tries a real YAML parse first; falls back to a small regex so extraction
    works even when PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError:
        # PyYAML is optional; the regex fallback below still works without it.
        yaml = None

    if yaml is not None:
        try:
            cfg = yaml.safe_load(config_bytes)
            tokenizer = (cfg or {}).get("tokenizer") or {}
            value = tokenizer.get("model") or tokenizer.get("dir")
            if isinstance(value, str):
                return value.strip()
        except (yaml.YAMLError, AttributeError, TypeError):
            # Malformed YAML or an unexpected schema: fall back to the regex.
            pass

    for line in config_bytes.decode("utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("model:"):
            candidate = stripped.split(":", 1)[1].strip().strip("'\"")
            if candidate.endswith(".model"):
                return candidate
    return None


def _find_vocab_member(archive: tarfile.TarFile) -> str | None:
    """Locate a vocabulary dump in the archive (``vocabulary.txt``/``vocab.txt``)."""
    names = archive.getnames()
    for name in names:
        base = Path(name).name
        if base in {"vocabulary.txt", "vocab.txt"}:
            return name
    return None


def extract_assets(
    nemo_path: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Extract ``tokenizer.model`` (+ ``vocabulary.txt`` when present) from a
    ``.nemo`` archive into ``output_dir`` without loading NeMo.

    Returns a mapping of asset kind -> written path.

    Raises:
        SystemExit: if the archive is invalid or contains no tokenizer.
    """
    try:
        with tarfile.open(nemo_path, "r:gz") as archive:
            return _extract_from_archive(archive, nemo_path, output_dir, force=force)
    except (tarfile.TarError, OSError) as e:
        raise SystemExit(
            f"ERROR: {nemo_path} is not a readable .nemo (gzipped tar) archive: {e}"
        ) from e


def _extract_from_archive(
    archive: tarfile.TarFile,
    nemo_path: Path,
    output_dir: Path,
    *,
    force: bool,
) -> dict[str, Path]:
    """Extract tokenizer/vocabulary members from an already-open archive."""
    tokenizer_member = _find_tokenizer_member(archive)
    if tokenizer_member is None:
        raise SystemExit(
            "ERROR: no SentencePiece tokenizer (*_tokenizer.model) found in "
            f"{nemo_path}. Cannot export the assets the CTC runtime needs."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Path] = {}

    tokenizer_dest = output_dir / TOKENIZER_FILENAME
    _write_member(archive, tokenizer_member, tokenizer_dest, force=force)
    assets["tokenizer_model"] = tokenizer_dest

    vocab_member = _find_vocab_member(archive)
    if vocab_member is not None:
        vocab_dest = output_dir / VOCAB_FILENAME
        _write_member(archive, vocab_member, vocab_dest, force=force)
        assets["vocabulary"] = vocab_dest

    return assets


def _write_member(
    archive: tarfile.TarFile,
    member_name: str,
    dest: Path,
    *,
    force: bool,
) -> None:
    """Extract one archive member to ``dest``, honoring ``--force``."""
    data = _member_reader(archive, member_name)
    if data is None:
        raise SystemExit(f"ERROR: archive member {member_name} is not readable.")
    _write_bytes(dest, data, force=force)


def _write_bytes(dest: Path, data: bytes, *, force: bool) -> None:
    """Write ``data`` to ``dest``, refusing to overwrite unless ``force``."""
    if dest.exists() and not force:
        raise SystemExit(
            f"ERROR: {dest} already exists. Re-run with --force to overwrite."
        )
    dest.write_bytes(data)


# --------------------------------------------------------------------------- #
# ONNX export (NeMo / torch — export-time only dependencies)
# --------------------------------------------------------------------------- #
def _import_export_deps() -> tuple[Any, Any]:
    """Import torch + NeMo lazily; raise a clear, actionable error if missing.

    Returns ``(torch, nemo_asr)``.
    """
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except ImportError as e:
        raise RuntimeError(
            "FastConformer ONNX export requires PyTorch and NeMo (export-time "
            "only — the munajjam runtime does not need them). Install with:\n"
            "  pip install torch torchaudio --index-url "
            "https://download.pytorch.org/whl/cpu\n"
            '  pip install "nemo_toolkit[asr]" onnx onnxruntime\n'
            f"Missing dependency: {e}"
        ) from e
    return torch, nemo_asr


def export_onnx_graphs(
    nemo_path: Path,
    output_dir: Path,
    *,
    opset: int = DEFAULT_OPSET,
    force: bool = False,
) -> dict[str, Path]:
    """Load the checkpoint with NeMo and export both ONNX graphs.

    Returns a mapping of graph kind -> written path.

    Raises:
        RuntimeError: if NeMo/torch are unavailable or the export fails.
    """
    torch, nemo_asr = _import_export_deps()

    try:
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
            str(nemo_path)
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to restore checkpoint {nemo_path} with NeMo. Is it a "
            "FastConformer hybrid RNNT/CTC checkpoint? "
            f"Original error: {e}"
        ) from e

    # The CTC head must be active, otherwise the RNNT decoder is exported.
    try:
        model.change_decoding_strategy(decoder_type="ctc")
    except Exception as e:
        raise RuntimeError(
            f"Failed to switch the checkpoint to the CTC decoding strategy: {e}"
        ) from e

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = nemo_path.stem
    graphs: dict[str, Path] = {}

    # 1) Stock NeMo export (mel-input; informational only, not loaded by the
    #    runtime). NeMo appends ".onnx" to the path it is given.
    stock_path = output_dir / f"{stem}{STOCK_ONNX_SUFFIX}"
    if stock_path.exists() and not force:
        raise SystemExit(
            f"ERROR: {stock_path} already exists. Re-run with --force to overwrite."
        )
    stock_base = stock_path.with_suffix("")
    try:
        model.export(str(stock_base))
    except Exception as e:
        raise RuntimeError(f"NeMo stock export failed: {e}") from e
    if not stock_path.is_file():
        raise RuntimeError(f"NeMo stock export did not produce {stock_path}.")
    graphs["stock"] = stock_path

    # 2) Production raw-audio export: preprocessor -> encoder -> CTC head in a
    #    single traced graph. Signatures follow NeMo 2.0.0rc1, verified against
    #    the reference checkpoint in docs/fastconformer-onnx-validation.md.
    raw_onnx = output_dir / f"{stem}{RAW_AUDIO_ONNX_SUFFIX}"
    _write_bytes(raw_onnx, b"", force=force)  # honor --force / refuse overwrite

    class RawAudioFastConformer(torch.nn.Module):
        """Trace wrapper: NeMo preprocessor -> encoder -> CTC decoder.

        Inputs match the production contract: a raw 16 kHz float32 waveform
        and an int32 length; outputs are the CTC log-probabilities and the
        encoded length.
        """

        def __init__(self, traced: Any) -> None:
            super().__init__()
            self.preprocessor = traced.preprocessor
            self.encoder = traced.encoder
            self.ctc_decoder = traced.aux_ctc.decoder

        def forward(
            self,
            input_signal: Any,
            input_signal_length: Any,
        ) -> tuple[Any, Any]:
            processed, processed_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )
            encoded, encoded_length = self.encoder(
                audio_signal=processed, length=processed_length
            )
            logprobs, encoded_lengths = self.ctc_decoder(
                encoder_output=encoded, encoded_lengths=encoded_length
            )
            return logprobs, encoded_lengths

    wrapper = RawAudioFastConformer(model)
    dummy_signal = torch.randn(1, SAMPLE_RATE, dtype=torch.float32)
    dummy_length = torch.tensor([SAMPLE_RATE], dtype=torch.int32)
    try:
        torch.onnx.export(
            wrapper,
            (dummy_signal, dummy_length),
            str(raw_onnx),
            input_names=[INPUT_SIGNAL_NAME, INPUT_LENGTH_NAME],
            output_names=[OUTPUT_LOGPROBS_NAME, OUTPUT_LENGTH_NAME],
            dynamic_axes={
                INPUT_SIGNAL_NAME: {0: "batch", 1: "time"},
                INPUT_LENGTH_NAME: {0: "batch"},
                OUTPUT_LOGPROBS_NAME: {0: "batch", 1: "time"},
                OUTPUT_LENGTH_NAME: {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Raw-audio ONNX export failed for {raw_onnx}: {e}"
        ) from e
    graphs["raw_audio"] = raw_onnx

    return graphs


# --------------------------------------------------------------------------- #
# Post-export validation (onnxruntime)
# --------------------------------------------------------------------------- #
def validate_onnx(raw_onnx: Path) -> None:
    """Run a tiny sanity inference on the exported raw-audio graph.

    Checks the I/O contract before running: input/output tensor names and
    ``NodeArg.type`` metadata (float32 waveform + int32 length inputs,
    float32 ``logprobs`` + int64 ``encoded_lengths`` outputs), then runs a
    short inference and validates the output shapes, including the trailing
    blank class (V+1 = 1025 for the reference model).

    Raises:
        SystemExit: if the graph does not satisfy the expected contract.
    """
    try:
        import numpy as np
        import onnxruntime
    except ImportError as e:
        raise SystemExit(
            "ERROR: --validate requires onnxruntime and numpy "
            f"(install with: pip install onnxruntime numpy): {e}"
        ) from e

    try:
        session = onnxruntime.InferenceSession(
            str(raw_onnx), providers=["CPUExecutionProvider"]
        )
        inputs = {i.name: i for i in session.get_inputs()}
        if set(inputs) != {INPUT_SIGNAL_NAME, INPUT_LENGTH_NAME}:
            raise SystemExit(
                "ERROR: unexpected graph inputs "
                f"{sorted(inputs)}; expected {INPUT_SIGNAL_NAME} and "
                f"{INPUT_LENGTH_NAME}."
            )

        outputs_meta = {o.name: o for o in session.get_outputs()}
        if set(outputs_meta) != {OUTPUT_LOGPROBS_NAME, OUTPUT_LENGTH_NAME}:
            raise SystemExit(
                "ERROR: unexpected graph outputs "
                f"{sorted(outputs_meta)}; expected {OUTPUT_LOGPROBS_NAME} and "
                f"{OUTPUT_LENGTH_NAME}."
            )

        # Validate the documented I/O dtype contract (NodeArg.type) BEFORE
        # running inference, so a dtype mismatch is reported with an actionable
        # message naming the tensor, its actual type and the expected type
        # instead of a generic runtime feed error.
        nodes = {**inputs, **outputs_meta}
        for name, expected in EXPECTED_IO_TYPES.items():
            actual = str(getattr(nodes[name], "type", ""))
            if actual != expected:
                raise SystemExit(
                    f"ERROR: unexpected type for ONNX tensor {name!r}: "
                    f"got {actual!r}, expected {expected!r}."
                )

        signal = np.zeros((1, 1600), dtype=np.float32)  # 0.1 s @ 16 kHz
        length = np.array([1600], dtype=np.int32)
        outputs = session.run(None, {INPUT_SIGNAL_NAME: signal, INPUT_LENGTH_NAME: length})
        result = {o.name: a for o, a in zip(session.get_outputs(), outputs, strict=True)}

        logprobs = result[OUTPUT_LOGPROBS_NAME]
        encoded_lengths = result[OUTPUT_LENGTH_NAME]
        if logprobs.ndim != 3 or logprobs.shape[0] != 1 or logprobs.shape[2] <= 1:
            raise SystemExit(
                f"ERROR: unexpected logprobs shape {logprobs.shape}; expected "
                "[1, T', V+1] with V+1 >= 2."
            )
        if np.asarray(encoded_lengths).reshape(-1).size != 1:
            raise SystemExit(
                f"ERROR: unexpected encoded_lengths shape {encoded_lengths.shape}."
            )
        print(
            f"Validation OK: {raw_onnx} inputs={sorted(inputs)} "
            f"logprobs={logprobs.shape} encoded_lengths="
            f"{np.asarray(encoded_lengths).reshape(-1).tolist()}"
        )
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"ERROR: ONNX validation failed: {e}") from e


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """Run the export (importable so unit tests can call it directly)."""
    args = parse_args(argv)

    checkpoint: Path = args.checkpoint
    if not checkpoint.is_file():
        print(
            f"ERROR: checkpoint not found: {checkpoint}\n"
            "Download it first, e.g.:\n"
            "  wget -O .model_validation/stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo "
            "https://huggingface.co/nvidia/"
            "stt_ar_fastconformer_hybrid_large_pc_v1.0/resolve/main/"
            "stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo",
            file=sys.stderr,
        )
        return 2

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    assets = extract_assets(checkpoint, output_dir, force=args.force)
    print(f"Extracted tokenizer: {assets['tokenizer_model']}")
    if "vocabulary" in assets:
        print(f"Extracted vocabulary: {assets['vocabulary']}")

    try:
        graphs = export_onnx_graphs(
            checkpoint, output_dir, opset=args.opset, force=args.force
        )
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Stock export (mel-input, informational): {graphs['stock']}")
    print(f"Production raw-audio export: {graphs['raw_audio']}")

    if not args.no_validate:
        validate_onnx(graphs["raw_audio"])

    print(
        "Done. For automatic server pickup, either point "
        "MUNAJJAM_FASTCONFORMER_MODEL_PATH / "
        "MUNAJJAM_FASTCONFORMER_TOKENIZER_MODEL_PATH at these files, or copy "
        "them into the provisioning cache directory (see "
        "docs/fastconformer-onnx-validation.md)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

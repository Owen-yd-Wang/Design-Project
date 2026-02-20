import os
import re
import csv
import argparse
from pathlib import Path
import numpy as np
from onnx import numpy_helper
import onnx 

DTYPE_MAP = {
    np.dtype("int8"):   "int8",
    np.dtype("uint8"):  "uint8",
    np.dtype("int16"):  "int16",
    np.dtype("uint16"): "uint16",
    np.dtype("int32"):  "int32",
    np.dtype("uint32"): "uint32",
    np.dtype("float16"):"float16",
    np.dtype("float32"):"float32",
}

def safe_name(name: str) -> str:
    """Make a filesystem-friendly filename from an ONNX tensor name."""
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return s[:180] if len(s) > 180 else s

def to_uint_le_words(raw: bytes, word_bytes: int) -> np.ndarray:
    """
    Pack raw bytes into little-endian words (word_bytes bytes per word),
    returned as uint array with one word per element.
    """
    if word_bytes == 1:
        return np.frombuffer(raw, dtype=np.uint8)

    # Pad to word boundary
    pad = (-len(raw)) % word_bytes
    if pad:
        raw = raw + b"\x00" * pad

    # Interpret as little-endian unsigned integers of width word_bytes
    if word_bytes == 2:
        return np.frombuffer(raw, dtype="<u2")
    if word_bytes == 4:
        return np.frombuffer(raw, dtype="<u4")
    if word_bytes == 8:
        return np.frombuffer(raw, dtype="<u8")

    raise ValueError("word_bytes must be 1,2,4, or 8")

def write_memh(path: Path, words: np.ndarray, word_bytes: int) -> None:
    """
    Write one hex word per line (compatible with $readmemh).
    """
    hex_width = word_bytes * 2  # bytes -> hex chars
    with path.open("w", encoding="utf-8") as f:
        for w in words:
            f.write(f"{int(w):0{hex_width}x}\n")

def main():
    ap = argparse.ArgumentParser(description="Extract ONNX initializers to Verilog .mem files ($readmemh).")
    ap.add_argument("--onnx", default="tiny-cnn-basys3\\\\tiny_cnn_cifar10_int8.onnx", help="Path to ONNX model")
    ap.add_argument("--out", default="tiny_onnx_mems", help="Output directory for .mem files")
    ap.add_argument("--word-bytes", type=int, default=1, choices=[1,2,4,8],
                    help="Bytes per line in .mem (1 for int8/uint8 weights; 4 for float32/int32, etc.)")
    ap.add_argument("--only-int8-weights", action="store_true",
                    help="If set, only dump int8/uint8 initializers (often the quantized weights).")
    args = ap.parse_args()

    onnx_path = Path(args.onnx).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model (handles external data if .onnx.data is present alongside)
    model = onnx.load(str(onnx_path))

    initializers = list(model.graph.initializer)
    if not initializers:
        raise RuntimeError("No initializers found in ONNX. (Is this model using external initializers not loaded?)")

    manifest_path = out_dir / "manifest.csv"
    dumped = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["tensor_name", "dtype", "shape", "num_elements", "raw_bytes",
                         "word_bytes", "mem_words", "mem_file"])

        for init in initializers:
            arr = numpy_helper.to_array(init)  # numpy array
            dtype = arr.dtype
            dtype_str = DTYPE_MAP.get(dtype, str(dtype))
            shape = list(arr.shape)

            if args.only_int8_weights and dtype not in (np.int8, np.uint8):
                continue

            # Raw byte representation of the tensor data, as stored in memory
            raw = arr.tobytes(order="C")  # row-major
            words = to_uint_le_words(raw, args.word_bytes)

            fname = safe_name(init.name) + f".w{args.word_bytes}.mem"
            mem_path = out_dir / fname
            write_memh(mem_path, words, args.word_bytes)

            writer.writerow([
                init.name,
                dtype_str,
                "x".join(map(str, shape)) if shape else "scalar",
                int(arr.size),
                len(raw),
                args.word_bytes,
                int(words.size),
                fname
            ])
            dumped += 1

    print(f"Done. Dumped {dumped} tensors to: {out_dir}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()

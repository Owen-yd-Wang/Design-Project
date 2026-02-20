#!/usr/bin/env python3
"""
Test Image Preparation Script for Quantized MobileNetV2 Testbench

This script helps prepare test images for the SystemVerilog testbench by:
1. Loading an image file (PNG, JPG, etc.)
2. Resizing to 32×32×3
3. Quantizing to INT8 [0-255]
4. Exporting in hex format (.mem)

Usage:
    python prepare_testbench_image.py input.jpg output.mem
    python prepare_testbench_image.py image.png --output test_image.mem
    python prepare_testbench_image.py dataset/ --batch  # Process multiple images
"""

import sys
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import struct


def load_image(image_path):
    """Load and decode image file."""
    try:
        img = Image.open(image_path)
        return np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def resize_image(img, size=(32, 32)):
    """Resize image to target dimensions."""
    if len(img.shape) == 2:  # Grayscale
        img = np.stack([img] * 3, axis=-1)
    
    # Convert to PIL, resize, convert back
    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img = pil_img.resize(size, Image.BILINEAR)
    
    return np.array(pil_img, dtype=np.float32)


def normalize_image(img, mean=None, std=None):
    """Normalize image to [0, 1] or using provided mean/std."""
    if mean is None:
        # Simple min-max normalization
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
    else:
        # Subtract mean and divide by std
        img = (img - np.array(mean)) / np.array(std)
        img = (img + 1) / 2  # Bring to [0, 1] range
    
    return img


def quantize_to_int8(img, scale=255.0):
    """Quantize normalized image to INT8 [0, 255]."""
    # Ensure input is normalized to [0, 1]
    img = np.clip(img, 0.0, 1.0)
    
    # Scale to [0, 255]
    img_int8 = (img * scale).astype(np.uint8)
    
    return img_int8


def export_hex_mem(img, output_path, verbose=False):
    """Export quantized int8 image in hex format (.mem)."""
    # Ensure shape is (H, W, C) and flatten
    if img.ndim == 3:
        H, W, C = img.shape
        # Flatten in H-W-C order (row-major for spatial, then channels)
        pixels = img.reshape(-1)
    else:
        pixels = img.flatten()
    
    # Write hex format (one byte per line)
    with open(output_path, 'w') as f:
        for pixel in pixels:
            f.write(f"{int(pixel):02X}\n")
    
    if verbose:
        print(f"Exported {len(pixels)} bytes to {output_path}")
        print(f"File size: {len(pixels)} bytes")
        print(f"Image dimensions: {H}×{W}×{C}")
    
    return len(pixels)


def import_hex_mem(mem_path, expected_size=3072):
    """Import hex .mem file back to numpy array."""
    bytes_list = []
    
    try:
        with open(mem_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('/'):
                    try:
                        bytes_list.append(int(line, 16))
                    except ValueError:
                        pass
        
        return np.array(bytes_list, dtype=np.uint8)
    except Exception as e:
        print(f"Error reading {mem_path}: {e}")
        return None


def verify_mem_file(mem_path, expected_size=3072):
    """Verify .mem file is valid."""
    try:
        with open(mem_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('/')]
        
        # Check size
        if len(lines) == expected_size:
            print(f"✓ File size correct: {len(lines)} bytes (expected {expected_size})")
            return True
        else:
            print(f"✗ File size incorrect: {len(lines)} bytes (expected {expected_size})")
            return False
    except Exception as e:
        print(f"Error verifying {mem_path}: {e}")
        return False


def list_statistics(img):
    """Print image statistics."""
    print(f"\nImage Statistics:")
    print(f"  Shape: {img.shape}")
    print(f"  Data type: {img.dtype}")
    print(f"  Min value: {img.min()}")
    print(f"  Max value: {img.max()}")
    print(f"  Mean value: {img.mean():.2f}")
    print(f"  Std dev: {img.std():.2f}")


def process_image(input_path, output_path, norm_mean=None, norm_std=None, verbose=True):
    """Full pipeline: load → resize → normalize → quantize → export."""
    print(f"\nProcessing: {input_path}")
    
    # Load
    print("  [1] Loading image...", end="", flush=True)
    img = load_image(input_path)
    if img is None:
        return False
    print(f" ✓ ({img.shape})")
    
    # Resize
    print("  [2] Resizing to 32×32×3...", end="", flush=True)
    img = resize_image(img, (32, 32))
    print(f" ✓")
    
    # Normalize
    print("  [3] Normalizing to [0, 1]...", end="", flush=True)
    img = normalize_image(img, mean=norm_mean, std=norm_std)
    print(f" ✓")
    
    # Quantize
    print("  [4] Quantizing to INT8 [0, 255]...", end="", flush=True)
    img_int8 = quantize_to_int8(img)
    print(f" ✓")
    
    # Export
    print(f"  [5] Exporting to {output_path}...", end="", flush=True)
    num_bytes = export_hex_mem(img_int8, output_path, verbose=False)
    print(f" ✓")
    
    if verbose:
        list_statistics(img_int8)
        print(f"\nOutput file: {output_path} ({num_bytes} bytes)")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare test images for quantized MobileNetV2 testbench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python prepare_testbench_image.py input.jpg test_image.mem
  
  # With custom output path
  python prepare_testbench_image.py cat.png --output cat_test.mem
  
  # Batch process directory
  python prepare_testbench_image.py images/ --batch
  
  # Verify existing .mem file
  python prepare_testbench_image.py test_image.mem --verify
  
  # Process with ImageNet normalization
  python prepare_testbench_image.py img.jpg output.mem \\
    --mean 0.485 0.456 0.406 \\
    --std 0.229 0.224 0.225
        """
    )
    
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('-o', '--output', default='output.mem', 
                        help='Output .mem file path (default: output.mem)')
    parser.add_argument('-b', '--batch', action='store_true',
                        help='Batch process all images in directory')
    parser.add_argument('--verify', action='store_true',
                        help='Verify .mem file instead of processing')
    parser.add_argument('--mean', type=float, nargs=3,
                        help='RGB mean values for normalization')
    parser.add_argument('--std', type=float, nargs=3,
                        help='RGB std dev values for normalization')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Verify mode
    if args.verify:
        print(f"\nVerifying: {args.input}")
        if verify_mem_file(args.input):
            print("✓ File is valid!")
            return 0
        else:
            print("✗ File has issues!")
            return 1
    
    # Batch mode
    if args.batch:
        input_dir = Path(args.input)
        if not input_dir.is_dir():
            print(f"Error: {args.input} is not a directory")
            return 1
        
        img_files = list(input_dir.glob("*.jpg")) + \
                   list(input_dir.glob("*.png")) + \
                   list(input_dir.glob("*.jpeg"))
        
        if not img_files:
            print(f"No image files found in {args.input}")
            return 1
        
        print(f"Found {len(img_files)} images")
        success_count = 0
        
        for img_path in img_files:
            output_mem = img_path.stem + ".mem"
            if process_image(str(img_path), output_mem, 
                            args.mean, args.std, args.verbose):
                success_count += 1
        
        print(f"\n✓ Processed {success_count}/{len(img_files)} images")
        return 0 if success_count == len(img_files) else 1
    
    # Single image mode
    input_path = args.input
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist")
        return 1
    
    if process_image(input_path, args.output, args.mean, args.std, args.verbose):
        print(f"\n✓ Successfully created {args.output}")
        
        # Verify
        if verify_mem_file(args.output, expected_size=32*32*3):
            print("✓ Output file verified!")
        
        return 0
    else:
        print(f"\n✗ Failed to process {input_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Prepare egypt_cat.jpg for hardware simulation"""
import numpy as np
from PIL import Image

print("Loading egypt_cat.jpg...")
img = Image.open('egypt_cat.jpg')
print(f"Original size: {img.size}")

# Convert to RGB if needed
if img.mode != 'RGB':
    img = img.convert('RGB')

# Resize to 224x224
img = img.resize((32, 32), Image.Resampling.LANCZOS)
print(f"Resized to: {img.size}")

# Convert to numpy
img_array = np.array(img, dtype=np.uint8)
print(f"Array shape: {img_array.shape}")

# Save as .mem file (HWC order, one byte per line in hex)
print("\nGenerating test_image.mem...")
with open('test_image.mem', 'w') as f:
    for h in range(32):
        for w in range(32):
            for c in range(3):
                f.write(f"{img_array[h, w, c]:02x}\n")

print(f"Wrote {32*32*3} bytes to test_image.mem")

# Save as numpy
np.save('test_image.npy', img_array)
print("Saved test_image.npy")

# Show sample
print(f"\nSample pixels (top-left):")
for h in range(3):
    for w in range(3):
        print(f"  ({h},{w}): R={img_array[h,w,0]:3d} G={img_array[h,w,1]:3d} B={img_array[h,w,2]:3d}")

print("\n✅ Image preparation complete!")
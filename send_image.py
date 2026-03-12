"""
Send a CIFAR-10 image to Basys3 over UART and read back the predicted class.

Usage:
  python send_image.py --port COM3                     # first test image from CIFAR-10
  python send_image.py --port COM3 --index 5           # 6th test image
  python send_image.py --port COM3 --image cat.jpg     # any JPEG/PNG (resized to 32x32)

Requirements:  pip install pyserial pillow numpy
               (for CIFAR-10 loading: pip install torchvision torch --index-url https://download.pytorch.org/whl/cpu)
"""

import argparse
import time
import struct
import sys
import numpy as np

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_from_cifar10_binary(path, index=0):
    """Load from CIFAR-10 binary file (test_batch or data_batch_*)."""
    import pickle
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    data   = d[b'data']    # shape (N, 3072), CHW order
    labels = d[b'labels']
    img_chw = data[index]  # already uint8 CHW: R[0..1023], G[1024..2047], B[2048..3071]
    return img_chw, labels[index]


def load_from_torchvision(index=0):
    """Download CIFAR-10 test set via torchvision and return CHW bytes."""
    import torchvision
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    img_pil, label = testset[index]
    img_hwc = np.array(img_pil, dtype=np.uint8)          # (32,32,3)
    img_chw = img_hwc.transpose(2, 0, 1).flatten()        # -> CHW flat
    return img_chw, label


def load_from_image_file(path):
    """Load any image file, resize to 32x32, return CHW bytes (label unknown=-1)."""
    from PIL import Image
    img = Image.open(path).convert('RGB').resize((32, 32), Image.LANCZOS)
    img_hwc = np.array(img, dtype=np.uint8)
    img_chw = img_hwc.transpose(2, 0, 1).flatten()
    return img_chw, -1


def get_image(args):
    """Return (img_chw_uint8 [3072], true_label_or_-1)."""
    if args.image:
        return load_from_image_file(args.image)

    # Try CIFAR-10 binary file first
    candidates = [
        'cifar-10-batches-py/test_batch',
        'data/cifar-10-batches-py/test_batch',
        'test_batch',
    ]
    for c in candidates:
        try:
            return load_from_cifar10_binary(c, args.index)
        except FileNotFoundError:
            pass

    # Fall back to torchvision (will download ~170 MB on first run)
    print("CIFAR-10 binary not found locally — downloading via torchvision...")
    try:
        return load_from_torchvision(args.index)
    except ImportError:
        sys.exit("ERROR: install torchvision or provide --image <file>")


# ---------------------------------------------------------------------------
# UART send / receive
# ---------------------------------------------------------------------------

def send_and_receive(port, baud, img_chw):
    import serial

    assert len(img_chw) == 3072, f"Expected 3072 bytes, got {len(img_chw)}"

    # img_chw is uint8; reinterpret as int8 bytes for transmission
    # (byte values are identical; the FPGA stores them as signed [7:0] but
    #  the MAC reads them as unsigned via {1'b0, rdata})
    payload = bytes(img_chw.astype(np.uint8))

    print(f"Opening {port} at {baud} baud ...")
    with serial.Serial(port, baud, timeout=10) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print(f"Sending 3072 bytes ...")
        t0 = time.time()
        ser.write(payload)
        ser.flush()

        print("Waiting for result ...")
        # Response: "Class: N\r\n" = 10 bytes
        response = ser.read(10)
        elapsed = time.time() - t0

    return response, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Send CIFAR-10 image to Basys3 CNN')
    parser.add_argument('--port',  required=True, help='Serial port, e.g. COM3')
    parser.add_argument('--baud',  type=int, default=115200)
    parser.add_argument('--index', type=int, default=0,
                        help='CIFAR-10 test image index (default 0)')
    parser.add_argument('--image', default=None,
                        help='Path to any image file (overrides --index)')
    args = parser.parse_args()

    # Load image
    img_chw, true_label = get_image(args)
    print(f"Image loaded: {len(img_chw)} bytes (CHW order)")
    if true_label >= 0:
        print(f"True label   : {true_label} ({CIFAR10_CLASSES[true_label]})")

    # Send
    response, elapsed = send_and_receive(args.port, args.baud, img_chw)

    # Parse response "Class: N\r\n"
    print(f"\nRaw response : {response!r}  ({elapsed:.2f}s)")
    try:
        text = response.decode('ascii').strip()
        predicted = int(text.split()[-1])
        print(f"Predicted    : {predicted} ({CIFAR10_CLASSES[predicted]})")
        if true_label >= 0:
            match = "CORRECT" if predicted == true_label else "WRONG"
            print(f"Result       : {match}")
    except Exception as e:
        print(f"Could not parse response: {e}")


if __name__ == '__main__':
    main()

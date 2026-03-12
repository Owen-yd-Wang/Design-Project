"""
Test FPGA inference accuracy over multiple CIFAR-10 images.
Usage: python test_accuracy.py
"""
import serial, time, pickle, numpy as np

CIFAR_PATH = 'cifar-10-batches-py/test_batch'
COM_PORT   = 'COM4'
N_TEST     = 500  # number of images to test

CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

def send_image(ser, img_uint8):
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    payload = bytes(img_uint8.astype('uint8'))
    ser.write(payload)
    ser.flush()
    response = ser.read(10)
    return response

def parse_class(response):
    """Parse b'Class: N\\r\\n' -> int N, or -1 on failure."""
    try:
        s = response.decode('ascii').strip()
        return int(s.split(':')[1].strip())
    except Exception:
        return -1

print(f"Loading CIFAR-10 test batch from {CIFAR_PATH} ...")
with open(CIFAR_PATH, 'rb') as f:
    d = pickle.load(f, encoding='bytes')

images = d[b'data'][:N_TEST]          # shape (N, 3072), uint8
labels = list(d[b'labels'][:N_TEST])

print(f"Testing {N_TEST} images on FPGA at {COM_PORT} ...")
print(f"{'Idx':>4}  {'True':>12}  {'Pred':>12}  {'OK':>3}  Raw")
print("-" * 55)

correct = 0
pred_counts = [0] * 10

with serial.Serial(COM_PORT, 115200, timeout=15) as ser:
    time.sleep(0.05)

    for i in range(N_TEST):
        img = images[i]
        true_cls = labels[i]

        resp = send_image(ser, img)
        pred_cls = parse_class(resp)

        ok = (pred_cls == true_cls)
        if ok:
            correct += 1
        if 0 <= pred_cls < 10:
            pred_counts[pred_cls] += 1

        true_name = CLASSES[true_cls]
        pred_name = CLASSES[pred_cls] if 0 <= pred_cls < 10 else '???'
        mark = '✓' if ok else '✗'
        print(f"{i:>4}  {true_name:>12}  {pred_name:>12}  {mark:>3}  {resp}")

        time.sleep(0.05)  # brief pause between images

print("-" * 55)
print(f"Accuracy: {correct}/{N_TEST} = {100*correct/N_TEST:.1f}%")
print(f"\nPrediction distribution:")
for c, cnt in enumerate(pred_counts):
    bar = '#' * cnt
    print(f"  {c} {CLASSES[c]:>12}: {cnt:>3}  {bar}")
print()
if max(pred_counts) == N_TEST:
    print("WARNING: FPGA predicts the SAME class for ALL images!")
    print("  → Weights almost certainly not loaded. Check synthesis log for:")
    print("    WARNING: [Synth 8-3332] Memory file not found")
elif correct == 0:
    print("WARNING: 0% accuracy — weights may be loaded but computation is wrong.")
else:
    print(f"Accuracy looks reasonable. Expected ~70% with correct weights.")

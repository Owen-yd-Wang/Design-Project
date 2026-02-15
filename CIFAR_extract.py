import pickle
from PIL import Image

CALIBRATION_PATH = "Calibration_Data/"
CIFAR_PATH = "cifar-10-batches-py/"

def CIFAR_extract(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

rs = CIFAR_extract(CIFAR_PATH +"data_batch_1")

def raw_byte_realign(data):
    result = []
    for i in range(len(data) // 3):
        result.append(data[i])
        result.append(data[i+1024])
        result.append(data[i+2048])
    return result

def raw_rgb_bytes_to_jpg(raw_bytes, width, height, out_path):
    expected = width * height * 3
    if len(raw_bytes) != expected:
        raise ValueError(f"Expected {expected} bytes, got {len(raw_bytes)}")
    img = Image.frombytes('RGB', (width, height), bytes(raw_bytes))
    img.save(out_path)





for i in range(300):
    data = raw_byte_realign(rs[b'data'][i])
    raw_rgb_bytes_to_jpg(bytes(data), 32, 32, CALIBRATION_PATH + rs[b'filenames'][i].decode("utf-8"))

print(rs.keys())
print(rs[b'data'][0])
print(len(rs[b'data'][0]))
print(type(rs[b'labels']))
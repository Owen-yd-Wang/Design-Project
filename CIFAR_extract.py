import pickle
from PIL import Image


def CIFAR_extract(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

CIFAR_path = "cifar-10-batches-py/"

rs = CIFAR_extract(CIFAR_path+"data_batch_1")

def raw_byte_to_jpg(data):
    result = []
    for i in range(0,1023):
        result.append(data[i])
        result.append(data[i+1023])
        result.append(data[i])
    return result

def raw_rgb_bytes_to_jpg(raw_bytes, width, height, out_path):
    expected = width * height * 3

    if len(raw_bytes) != expected:
        raise ValueError(
            f"Wrong size: got {len(raw_bytes)}, expected {expected}"
        )

    img = Image.frombytes(
        "RGB",
        (width, height),
        raw_bytes
    )

    img.save(out_path, "JPEG", quality=95)



print(rs.keys())
print(rs[b'data'][0])
print(len(rs[b'data'][0]))
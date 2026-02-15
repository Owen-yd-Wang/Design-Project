from torchvision import models, datasets, transforms as TF
from PIL import Image
import numpy as np
import onnxruntime as OXR
import torch
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import os
import CIFAR_extract

CALIBRATION_PATH = "Calibration_Data/"

def main():
    # Pretrained MobileNetV2 model as the CNN basis
    mobilenet_v2 = models.mobilenet_v2(pretrained=True)

    # Set detault image size
    image_height = 224
    image_width = 224
    x = torch.randn(1, 3, image_height, image_width, requires_grad=True)

    # Export the model
    torch.onnx.export(mobilenet_v2,            # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "mobilenet_v2_float.onnx", # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output']) # the model's output names

    # Preprocessing the image for ONNX runtime prep
    def preprocess_image(image_path, height, width, channels=3):
        image = Image.open(image_path)
        image = image.resize((width, height))
        image_data = np.asarray(image).astype(np.float32)
        image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
        mean = np.array([0.079, 0.05, 0]) + 0.406
        std = np.array([0.005, 0, 0.001]) + 0.224
        for channel in range(image_data.shape[0]):
            image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
        image_data = np.expand_dims(image_data, 0)
        return image_data

    # Read ImageNet categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    session_fp32 = OXR.InferenceSession("mobilenet_v2_float.onnx")

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def run_sample(session, image_file, categories):
        output = session.run([], {'input':preprocess_image(image_file, image_height, image_width)})[0]
        output = output.flatten()
        output = softmax(output) # this is optional
        top5_catid = np.argsort(-output)[:5]
        for catid in top5_catid:
            print(categories[catid], output[catid])

    def preprocess_func(images_folder, height, width, size_limit=0):
        image_names = os.listdir(images_folder)
        if size_limit > 0 and len(image_names) >= size_limit:
            batch_filenames = [image_names[i] for i in range(size_limit)]
        else:
            batch_filenames = image_names
        unconcatenated_batch_data = []

        for image_name in batch_filenames:
            image_filepath = images_folder + '/' + image_name
            image_data = preprocess_image(image_filepath, height, width)
            unconcatenated_batch_data.append(image_data)
        batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
        return batch_data


    class MobilenetDataReader(CalibrationDataReader):
        def __init__(self, calibration_image_folder):
            self.image_folder = calibration_image_folder
            self.preprocess_flag = True
            self.enum_data_dicts = []
            self.datasize = 0

        def get_next(self):
            if self.preprocess_flag:
                self.preprocess_flag = False
                nhwc_data_list = preprocess_func(self.image_folder, image_height, image_width, size_limit=0)
                self.datasize = len(nhwc_data_list)
                self.enum_data_dicts = iter([{'input': nhwc_data} for nhwc_data in nhwc_data_list])
            return next(self.enum_data_dicts, None)

    dr = MobilenetDataReader(CALIBRATION_PATH)

    quantize_static('mobilenet_v2_float.onnx',
                    'mobilenet_v2_uint8.onnx',
                    dr)

    print('ONNX full precision model size (MB):', os.path.getsize("mobilenet_v2_float.onnx")/(1024*1024))
    print('ONNX quantized model size (MB):', os.path.getsize("mobilenet_v2_uint8.onnx")/(1024*1024))


    # Cat and cockroach example
    run_sample(session_fp32, 'egypt_cat.jpg', categories)
    run_sample(session_fp32, 'cockroach.jpg', categories)
    session_quant = OXR.InferenceSession("mobilenet_v2_uint8.onnx")
    run_sample(session_quant, 'egypt_cat.jpg', categories)
    run_sample(session_quant, 'cockroach.jpg', categories)
    
if __name__ == "__main__":
    main()
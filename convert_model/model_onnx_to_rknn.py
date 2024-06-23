import os
import sys
from rknn.api import RKNN 
import onnx

model = onnx.load("/home/fumi/convert_model_rknn/rknn-toolkit2/policy.onnx")

output_file = "rknn_model"

output_directory = '/home/fumi/convert_model_rknn/rknn-toolkit2/converted_model'

output_path = os.path.join(output_directory, output_file)

rknn = RKNN(verbose=False)

rknn.config(target_platform="rk3588S")


ret = rknn.load_onnx("/home/fumi/convert_model_rknn/rknn-toolkit2/policy.onnx")

ret = rknn.build(do_quantization=False)

ret = rknn.export_rknn("what_the_sigma.rknn")
                       
rknn.release()

import os
import sys
from rknn.api import RKNN 
import onnx

model = onnx.load("/home/fumi/CodeStuff/Mistletoe-Sim/quadruped_state_estimator/state_estimator.onnx")

output_file = "rknn_model"

output_directory = '/home/fumi/CodeStuff/Mistletoe-Sim/quadruped_state_estimator'

output_path = os.path.join(output_directory, output_file)

rknn = RKNN(verbose=False)

rknn.config(target_platform="rk3588S")


ret = rknn.load_onnx("/home/fumi/CodeStuff/Mistletoe-Sim/quadruped_state_estimator/state_estimator.onnx")

ret = rknn.build(do_quantization=False)

ret = rknn.export_rknn("/home/fumi/CodeStuff/Mistletoe-Sim/convert_model/output/state_estimator.rknn")
                       
rknn.release()

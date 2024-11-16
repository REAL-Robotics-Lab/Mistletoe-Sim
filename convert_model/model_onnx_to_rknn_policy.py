import os
import sys
from rknn.api import RKNN 
import onnx

model = onnx.load("/home/fumi/CodeStuff/Mistletoe-Sim/convert_model/policy.onnx")

output_file = "rknn_model"
rknn = RKNN(verbose=False)

rknn.config(target_platform="rk3588S")


ret = rknn.load_onnx("/home/fumi/CodeStuff/Mistletoe-Sim/convert_model/policy.onnx")

ret = rknn.build(do_quantization=False)

ret = rknn.export_rknn("/home/fumi/CodeStuff/Mistletoe-Sim/convert_model/output/policy.rknn")
                       
rknn.release()

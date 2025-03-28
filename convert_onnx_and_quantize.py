import os
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import torchvision.models as models
from multitask_model import MultiTaskModel

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.mobilenet_v2(pretrained=True)
num_classes = 2
num_frames = 5  # Number of frames to extract from each video
model = MultiTaskModel(model, num_frames=num_frames)
model.load_state_dict(torch.load('activity_detection_model.pth'))
model = model.to(device)
model.eval()

# Dummy input for the model
dummy_input = torch.randn(1, num_frames, 3, 224, 224).to(device)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "model.onnx", export_params=True, input_names=['input'], output_names=['binary_output', 'multi_output'])

# Load the ONNX model
model_path = 'model.onnx'
onnx_model = onnx.load(model_path)

# Quantize the model
quantized_model_path = 'model_quantized.onnx'
quantize_dynamic(model_path, quantized_model_path, weight_type=QuantType.QUInt8)
'''The function quantize_dynamic() applies dynamic quantization, which means that only the weights of the model are quantized to a lower precision
 (in this case, 8-bit unsigned integers (QUInt8)), while activations remain in their original precision and are quantized dynamically during inference.'''

print("Model has been converted to ONNX and quantized successfully.")
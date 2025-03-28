import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import onnxruntime as ort

# Paths
QUANTIZED_ONNX_MODEL_PATH = "model_quantized.onnx"
ACTIVITIES = ['Violence', 'Theft']  # Multi-class labels

# Load the ONNX model
ort_session = ort.InferenceSession(QUANTIZED_ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_video(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame {idx} from {video_path}")
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = transform(frame)
        frames.append(frame.unsqueeze(0))

    cap.release()
    return torch.cat(frames, dim=0).unsqueeze(0).numpy()  # Add batch dimension

def predict(video_path):
    # Preprocess the video
    input_tensor = preprocess_video(video_path)

    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)
    binary_output, multi_output = ort_outs

    # Determine if the activity is suspicious
    is_suspicious = binary_output[0][0] > 0.5

    if is_suspicious:
        # Get the probabilities and predicted class
        probabilities = torch.softmax(torch.tensor(multi_output[0]), dim=0).numpy()
        activity_index = np.argmax(probabilities)
        activity_name = ACTIVITIES[activity_index]
        probability_score = probabilities[activity_index] * 100  # Convert to percentage
        return f"Suspicious activity detected: {activity_name} (Confidence: {probability_score:.2f}%)"
    else:
        return "No suspicious activity detected."

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Video(),
    outputs="text",
    title="Activity Detection",
    description="Upload or record a video to detect suspicious activities with probability scores.",
)

if __name__ == "__main__":
    iface.launch()
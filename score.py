import os
import json
import base64
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# These must match how you saved and named your classes
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituiray_tumor"]
IMG_SIZE = 224

def init():
    """
    Load the model and set up preprocessing.
    Azure ML will call this once when the service starts.
    """
    global model, transform, device

    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Model architecture
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    # 3. Load weights
    # If deployed to Azure ML, AZUREML_MODEL_DIR points at the folder containing your model file.
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "saved_models", "resnet50_brain_tumor.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # 4. Preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])


def run(raw_data):
    """
    raw_data: JSON string with a single field "image" whose value
              is the base64-encoded JPEG bytes.
    Returns: JSON with fields "predicted_label" and "scores".
    """
    try:
        # 1. Decode input
        data = json.loads(raw_data)
        img_bytes = base64.b64decode(data["image"])
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        # 2. Preprocess & predict
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
            idx = int(torch.argmax(logits, dim=1))

        # 3. Prepare output
        result = {
            "predicted_label": CLASS_NAMES[idx],
            "scores": {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
        }
        return json.dumps(result)

    except Exception as e:
        # In Azure ML, returning a string here triggers a 500 with your error
        return json.dumps({"error": str(e)})

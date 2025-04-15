import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import json
from maml import ModelAgnosticMetaLearning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class mapping (folder_id -> human label)
with open("class_names.json", "r") as f:
    class_map = json.load(f)

# Image transform
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

def load_image(img_path):
    return transform(Image.open(img_path).convert("RGB"))

def prepare_data(images):
    return torch.stack([load_image(img) for img in images])

def predict(support_images, support_labels, query_image_path):
    """
    Few-shot prediction using MAML-style adaptation

    Args:
        support_images (list): Paths to support images (e.g., 5-shot support set)
        support_labels (list): Class indices corresponding to support images
        query_image_path (str): Path to a single query image
    Returns:
        str: Predicted human-readable class name
    """
    # Unique label count
    n_classes = len(set(support_labels))

    # === 1. Load Base Model ===
    base_model = ModelAgnosticMetaLearning(num_classes=24).to(device)
    # Load only encoder weights (ignore classifier layer)
    state_dict = torch.load("model/maml_final.pth", map_location=device)
    # Remove classifier weights if shape mismatch
    for key in list(state_dict.keys()):
        if key.startswith("classifier"):
            del state_dict[key]

    base_model.load_state_dict(state_dict, strict=False)

   # === 2. Clone for Fast Adaptation ===
    model = ModelAgnosticMetaLearning(num_classes=n_classes).to(device)

    # Load only encoder weights from base_model into model (ignore classifier)
    model_state = model.state_dict()
    base_state = base_model.state_dict()

    #Copy encoder weights only
    for key in base_state:
        if key.startswith("encoder."):
            model_state[key] = base_state[key]

    model.load_state_dict(model_state)

    # === 3. Prepare Inputs ===
    support_inputs = prepare_data(support_images).to(device)
    support_targets = torch.tensor(support_labels).to(device)
    query_input = load_image(query_image_path).unsqueeze(0).to(device)

    # === 4. Inner-loop adaptation ===
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    support_outputs = model(support_inputs)
    loss = criterion(support_outputs, support_targets)
    loss.backward()
    optimizer.step()

    # === 5. Predict ===
    model.eval()
    with torch.no_grad():
        query_output = model(query_input)
        predicted_idx = torch.argmax(query_output, dim=1).item()

    # === 6. Map class index to human label ===
    folder_names = sorted(class_map.keys())
    return class_map.get(folder_names[predicted_idx], "unknown")

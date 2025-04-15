import os
import json
import sys
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Allow import from models/
sys.path.append('./models')
from maml import ModelAgnosticMetaLearning

# === Dataset Loader ===
class FewShotDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

	# Load class name mapping
        json_path = os.path.join(os.path.dirname(__file__), "class_names.json")
        with open(json_path, "r") as f:
            self.class_names = json.load(f)

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        for cls in self.class_to_idx:
            cls_path = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# === Main Training ===
def main():
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])

    train_dataset = FewShotDataset(root_dir='dataset_split/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maml = ModelAgnosticMetaLearning().to(device)
    optimizer = torch.optim.Adam(maml.parameters(), lr=1e-3)

    for step, (inputs, targets) in enumerate(tqdm(train_loader)):
        if step >= 100:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        

        # Remap targets to 0..N-1
        unique_labels = torch.unique(targets)
        label_map = {label.item(): i for i, label in enumerate(unique_labels)}
        targets_mapped = torch.tensor([label_map[label.item()] for label in targets], dtype=torch.long).to(device)
        num_classes_in_batch = len(unique_labels)
        maml.classifier = nn.Linear(1600, num_classes_in_batch).to(device)
        outputs = maml(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets_mapped)
        loss.backward()
        optimizer.step()
        
        # Human-readable label preview
        if step % 10 == 0:
            folder_names = sorted(os.listdir('dataset_split/train'))
            json_path = os.path.join(os.path.dirname(__file__), "class_names.json")
            with open(json_path, "r") as f:
                class_map = json.load(f)
            readable_labels = [class_map.get(folder_names[i.item()], "unknown") for i in targets[:5]]
            print(f"Step {step + 1} — Sample classes: {readable_labels}")

    print(f"Step {step+1}/100 | Loss: {loss.item():.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(maml.state_dict(), "model/maml_final.pth")
    print("\n Training complete. Model saved to model/maml_final.pth")

if __name__ == '__main__':
    main()

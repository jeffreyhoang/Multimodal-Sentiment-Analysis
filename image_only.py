import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.models import resnet18, ResNet18_Weights

import re
import random
import copy
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

# =============================================================
# Set Random Seed
# =============================================================
SEED = 20
random.seed(SEED)


# =============================================================
# Load Data
# =============================================================
DATA_LABEL_PATH = 'MVSA_Single/labelResultAll.txt'
df = pd.read_csv(DATA_LABEL_PATH, sep=r'\t|,', engine='python')

print('Data loaded succesfully!')

# =============================================================
# Preprocess data
# =============================================================
df = df[df['text'] == df['image']]      # Filter inconsistent labels
df = df.drop(columns=['image'])     # Drop image label column
df = df.rename(columns={'text': 'label'})       # Rename label column

# Clean text data (not applied)
def clean_standard(text):
    text = re.sub(r"RT\s+@\w+:", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# Add text and images to df
texts = []
images = []

DATA_PATH = 'MVSA_Single/data'

for i in df['ID']:
    text_path = f'{DATA_PATH}/{i}.txt'
    img_path = f'{DATA_PATH}/{i}.jpg'

    with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().strip()
        texts.append(text)

    images.append(img_path)

df['text'] = texts
df['image'] = images

# Map labels to integers
id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
label2id = {'negative': 0, 'neutral': 1, 'positive': 2}

df['label'] = df['label'].map(label2id)

print('Data preprocessed succesfully!')


# =============================================================
# Split Data (70%-15%-15%)
# =============================================================
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=SEED,
    stratify=df['label']
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=SEED,
    stratify=temp_df['label']
)

print('Data split succesfully!')


# =============================================================
# Training Configuration
# =============================================================
NUM_CLASSES = 3
BATCH_SIZE = 16
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 8

FC_LR = 1e-4
LAYER_4LR = 1e-5
lAYER3_LR = 1e-6
LAYER2_LR = 1e-7

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Training configuration set up succesfully!')
print(f'Device: {DEVICE}')


# =============================================================
# Dataset Class
# =============================================================
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = str(self.image_paths[idx])
        label = self.labels[idx]

        img = Image.open(image_path).convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        return {
            'images': img,
            'labels': torch.tensor(label, dtype=torch.long)
        }


# =============================================================
# Image Transformations
# =============================================================
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# =============================================================
# Initialize Dataset and Dataloaders
# =============================================================
train_image_paths = train_df['image'].tolist()
val_image_paths = val_df['image'].tolist()
test_image_paths = test_df['image'].tolist()

train_labels = train_df['label'].tolist()
val_labels = val_df['label'].tolist()
test_labels = test_df['label'].tolist()

train_dataset = ImageDataset(train_image_paths, train_labels, train_transform)
val_dataset = ImageDataset(val_image_paths, val_labels, val_test_transform)
test_dataset = ImageDataset(test_image_paths, test_labels, val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print('Dataset and Dataloaders initialized succesfully!')


# =============================================================
# Text-Only Model Class
# =============================================================
class ResnetClassifier(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)

        self.model.fc = nn.Sequential(
        nn.Linear(self.model.fc.in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# =============================================================
# Train Function
# =============================================================
def train(model, data_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in data_loader:
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(x=images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    return avg_train_loss, train_acc


# =============================================================
# Evaluate Function for 1 Epoch
# =============================================================
def evaluate(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(x=images)

            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    val_precision = precision_score(all_true, all_preds, average='macro', zero_division=0)
    val_recall = recall_score(all_true, all_preds, average='macro', zero_division=0)
    val_f1 = f1_score(all_true, all_preds, average='macro', zero_division=0)


    return avg_val_loss, val_acc, val_precision, val_recall, val_f1

print('Train and evaluation functions initialized succesfully!')


# =============================================================
# Initialize Model
# =============================================================
model = ResnetClassifier(NUM_CLASSES, DROPOUT).to(DEVICE)

for param in model.model.parameters():
    param.requires_grad = False

for param in model.model.fc.parameters():
    param.requires_grad = True

for param in model.model.layer4.parameters():
    param.requires_grad = True

for param in model.model.layer3.parameters():
    param.requires_grad = True

torch.cuda.empty_cache()        # Clear GPU cache

optimizer = AdamW([     # Initialize optimizer
    {'params': model.model.fc.parameters(), 'lr': FC_LR},
    {'params': model.model.layer4.parameters(), 'lr': LAYER_4LR},
    {'params': model.model.layer3.parameters(), 'lr': lAYER3_LR} 
], weight_decay=WEIGHT_DECAY)

criterion = nn.CrossEntropyLoss()       # Initialize loss function

print('Model configuration initialized succesfully!\n')


# =============================================================
# Training Loop
# =============================================================
best_val_f1 = -1.0
best_epoch = -1
best_model_state = None

print('Training loop started:')
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion, DEVICE)
    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}')
    print()

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1
        best_model_state = copy.deepcopy(model.state_dict())

print('Training loop ended.\n')

# =============================================================
# Evaluate on Test Data
# =============================================================
model.load_state_dict(best_model_state)
test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion, DEVICE)
print('Image-only model evaluation on test set:')
print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f} | Test F1: {test_f1:.4f}')

# =============================================================
# Save Best Model
# =============================================================
torch.save(best_model_state, 'best_image_model.pth')

print('Best model saved successfully!')
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.models import resnet18, ResNet18_Weights

from transformers import AutoModel, AutoTokenizer

import re
import random
import copy
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

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
# Filter inconsistent labels
df = df[df['text'] == df['image']] 

# Drop image label column
df = df.drop(columns=['image'])     

# Rename label column
df = df.rename(columns={'text': 'label'})       

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
TEXT_MODEL_NAME = 'bert-base-uncased'
SEQUENCE_LENGTH = 256
TOKENIZER = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

NUM_CLASSES = 3
BATCH_SIZE = 32
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
PROJ_DIM = 256
NUM_EPOCHS = 8

BERT_LR = 3e-5
FC_LR = 1e-4
LAYER4_LR = 1e-5
LAYER3_LR = 1e-6
PROJ_LR = 1e-4
MULTIMODAL_CLASSIFIER_LR = 1e-4
GATE_LR = 5e-4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Training configuration set up succesfully!')
print(f'Device: {DEVICE}')


# =============================================================
# Dataset Class
# =============================================================
class TextImageDataset(Dataset):
    def __init__(self, texts, image_paths, labels, tokenizer, max_len, transforms):
        self.texts = texts
        self.image_paths = image_paths
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        image_path = str(self.image_paths[idx])
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        # Load image
        img = Image.open(image_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'image': img,
            'label': torch.tensor(label, dtype=torch.long)
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
train_texts = train_df['text'].tolist()
val_texts = val_df['text'].tolist()
test_texts = test_df['text'].tolist()

train_image_paths = train_df['image'].tolist()
val_image_paths = val_df['image'].tolist()
test_image_paths = test_df['image'].tolist()

train_labels = train_df['label'].tolist()
val_labels = val_df['label'].tolist()
test_labels = test_df['label'].tolist()

train_dataset = TextImageDataset(train_texts, train_image_paths, train_labels, TOKENIZER, SEQUENCE_LENGTH, train_transform)
val_dataset = TextImageDataset(val_texts, val_image_paths, val_labels, TOKENIZER, SEQUENCE_LENGTH, val_test_transform)
test_dataset = TextImageDataset(test_texts, test_image_paths, test_labels, TOKENIZER, SEQUENCE_LENGTH, val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print('Dataset and Dataloaders initialized succesfully!')


# =============================================================
# Vector-Gated Fusion Class
# =============================================================
class VectorGatedFusionClassifier(nn.Module):
    def __init__(self, text_model_name, num_classes, dropout, proj_dim):
        super().__init__()

        # Text encoder
        self.bert = AutoModel.from_pretrained(text_model_name)
        text_dim = self.bert.config.hidden_size

        # Image encoder
        self.weights = ResNet18_Weights.DEFAULT
        self.resnet = resnet18(weights=self.weights)
        image_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Projection layers
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.image_proj = nn.Linear(image_dim, proj_dim)

        # Vector gate
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, proj_dim)
        )

        # Fusion head
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.LayerNorm(proj_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, num_classes)
        )
  
    def forward(self, input_ids, attention_mask, image):
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_cls_output = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_proj(text_cls_output)

        image_outputs = self.resnet(image)
        image_features = self.image_proj(image_outputs)

        # Vector gate
        combined = torch.cat([text_features, image_features], dim=1)
        gate = torch.sigmoid(self.gate(combined))

        self.last_gate = gate.detach()

        # Fuse features
        fused_features = text_features * gate + image_features * (1 - gate)

        return self.classifier(fused_features)
    

# =============================================================
# Train Function
# =============================================================
def train(model, data_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)

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
# Evaluate Function
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)

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
model = VectorGatedFusionClassifier(TEXT_MODEL_NAME, NUM_CLASSES, DROPOUT, PROJ_DIM).to(DEVICE)

for param in model.parameters():
    param.requires_grad = False

for param in model.bert.parameters():
    param.requires_grad = True

for param in model.resnet.layer4.parameters():
    param.requires_grad = True

for param in model.resnet.layer3.parameters():
    param.requires_grad = True

for param in model.text_proj.parameters():
    param.requires_grad = True

for param in model.image_proj.parameters():
    param.requires_grad = True

for param in model.gate.parameters():
  param.requires_grad = True

for param in model.classifier.parameters():
  param.requires_grad = True

# Clear GPU cache
torch.cuda.empty_cache()

# Optimizer configuration
optimizer = AdamW([
    {'params': model.bert.parameters(), 'lr': BERT_LR, 'weight_decay': WEIGHT_DECAY},
    {'params': model.resnet.layer4.parameters(), 'lr': LAYER4_LR, 'weight_decay': WEIGHT_DECAY},
    {'params': model.resnet.layer3.parameters(), 'lr': LAYER3_LR, 'weight_decay': WEIGHT_DECAY},
    {'params': model.text_proj.parameters(), 'lr': PROJ_LR, 'weight_decay': WEIGHT_DECAY},
    {'params': model.image_proj.parameters(), 'lr': PROJ_LR, 'weight_decay': WEIGHT_DECAY},
    {'params': model.gate.parameters(), 'lr': GATE_LR, 'weight_decay': WEIGHT_DECAY},
    {'params': model.classifier.parameters(), 'lr': MULTIMODAL_CLASSIFIER_LR, 'weight_decay': WEIGHT_DECAY},
    ])

# Initialize loss function
criterion = nn.CrossEntropyLoss()


# =============================================================
# Training Loop
# =============================================================
best_val_f1 = -1.0
best_epoch = -1
best_model_state = None

print('Training loop started:')

for epoch in range(NUM_EPOCHS):
  print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
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
print('Vector-gated fusion model evaluation on test set:')
print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f} | Test F1: {test_f1:.4f}')

# =============================================================
# Save Best Model
# =============================================================
torch.save(best_model_state, 'best_vector_gated_model.pth')

print('Best model saved successfully!')
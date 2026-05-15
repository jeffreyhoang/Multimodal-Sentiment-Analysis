import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

import re
import random
import copy
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

print('Dependencies imported succesfully!')

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
BERT_MODEL_NAME = 'bert-base-uncased'
NUM_CLASSES = 3
MAX_LENGTH = 256
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
NUM_EPOCHS = 8

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

print('Training configuration set up succesfully!')
print(f'Device: {DEVICE}')


# =============================================================
# Dataset Class
# =============================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# =============================================================
# Initialize Dataset and Dataloaders
# =============================================================
train_texts = train_df['text'].tolist()
val_texts = val_df['text'].tolist()
test_texts = test_df['text'].tolist()

train_labels = train_df['label'].tolist()
val_labels = val_df['label'].tolist()
test_labels = test_df['label'].tolist()

train_dataset = TextDataset(train_texts, train_labels, TOKENIZER, max_len=MAX_LENGTH)
val_dataset = TextDataset(val_texts, val_labels, TOKENIZER, max_len=MAX_LENGTH)
test_dataset = TextDataset(test_texts, test_labels, TOKENIZER, max_len=MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print('Dataset and Dataloaders initialized succesfully!')


# =============================================================
# Text-Only Model Class
# =============================================================
class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)

        return logits


# =============================================================
# Train Function
# =============================================================
def train(model, data_loader, criterion, optimizer, scheduler, device):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
    
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
        loss = criterion(outputs, labels)
    
        loss.backward()
        optimizer.step()
        scheduler.step()
    
        train_loss += loss.item() * input_ids.size(0)
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
    all_true= []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
    
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * input_ids.size(0)
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
model = BERTClassifier(BERT_MODEL_NAME, NUM_CLASSES, DROPOUT).to(DEVICE)

# Clear GPU cache
torch.cuda.empty_cache()

# Initialize optimizer
optimizer = AdamW(    
    model.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

 # Initialize loss function
criterion = nn.CrossEntropyLoss()      

# Initialize learning rate scheduler
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(        
    optimizer, num_warmup_steps=0, 
    num_training_steps=total_steps
)

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
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, DEVICE)
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
print('Text-only model evaluation on test set:')
print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f} | Test F1: {test_f1:.4f}')


# =============================================================
# Save Best Model
# =============================================================
torch.save(best_model_state, 'best_text_model.pth')

print('Best model saved successfully!')
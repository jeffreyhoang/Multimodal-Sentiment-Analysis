# Unimodal vs. Multimodal Sentiment Analysis Using Text and Image Data

This project uses the **MVSA-Single** (Multi-View Sentiment Analysis) dataset, which contains social media posts pairing short text captions with images, each labeled with a single sentiment: **positive**, **neutral**, or **negative**. We compare unimodal approaches (text-only, image-only) against multimodal fusion approaches for sentiment analysis. Multimodal models combine both modalities using fusion techniques including **score-level**, **embedding-level**, **scalar-gated**, and **vector-gated** fusion.

---

## Tech Stack

| Category        | Tools / Libraries                                      |
|-----------------|--------------------------------------------------------|
| Language        | Python                                                 |
| Deep Learning   | PyTorch, Torchvision                                   |
| NLP / Text      | Hugging Face Transformers (`bert-base-uncased`)        |
| Vision / Image  | ResNet-18 (pretrained on ImageNet via Torchvision)     |
| Data Processing | Pandas, Scikit-learn, Pillow                           |
| Hardware        | Google Colab, NVIDIA A100 GPU                          |

---

## Models

### BERT (Text)

The text encoder uses `bert-base-uncased` loaded via the Hugging Face `transformers` library. The `[CLS]` token representation from the final hidden state is passed through a dropout layer and a single linear classification head to produce logits over the three sentiment classes. All BERT parameters are fine-tuned end-to-end using AdamW with a linear learning rate schedule and warmup.

- **Optimizer:** AdamW (lr=3e-5, weight decay=1e-4)
- **Scheduler:** Linear decay with warmup
- **Max token length:** 256
- **Epochs:** 8

### ResNet-18 (Image)

The image encoder uses a pretrained ResNet-18 (ImageNet weights). The original fully connected head is replaced with a custom classifier: `Linear(512→256) → BatchNorm → ReLU → Dropout → Linear(256→3)`. To prevent overfitting, only the final classifier, layer4, and layer3 are unfrozen, with discriminative learning rates applied per layer group.

- **Optimizer:** AdamW with layer-wise learning rates
  - FC head: lr=1e-4
  - Layer 4: lr=1e-5
  - Layer 3: lr=1e-6
- **Weight decay:** 1e-4
- **Epochs:** 8
- **Image size:** 224×224 (ResizedCrop + HorizontalFlip + Rotation for training; CenterCrop for val/test)
- **Normalization:** ImageNet mean/std

---

## Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

Download the MVSA-Single dataset and place the extracted folder in the project root directory: https://www.kaggle.com/datasets/vincemarcs/mvsasingle

---

## Directory Structure

```text
Multimodal-Sentiment-Analysis/
│
├── MVSA_Single/               # Dataset folder (download separately)
├── train_text.py              # Text-only model
├── train_image.py             # Image-only model
├── train_score_fusion.py      # Score-level fusion
├── train_embedding_fusion.py  # Embedding-level fusion
├── train_scalar_gated.py      # Scalar-gated fusion
├── train_vector_gated.py      # Vector-gated fusion
└── requirements.txt
```

---

## Run Experiments

### Text-Only Model

```bash
python train_text.py
```

### Image-Only Model

```bash
python train_image.py
```

### Score-Level Fusion

```bash
python train_score_fusion.py
```

### Embedding-Level Fusion

```bash
python train_embedding_fusion.py
```

### Scalar-Gated Fusion

```bash
python train_scalar_gated.py
```

### Vector-Gated Fusion

```bash
python train_vector_gated.py
```

---

## Results

Overall performance comparison of all models on the MVSA-Single test set:

| Model                  | Accuracy   | Precision  | Recall     | F1-Score   |
|------------------------|------------|------------|------------|------------|
| Text Only              | 0.7738     | 0.7405     | 0.7266     | 0.7313     |
| Image Only             | 0.6504     | 0.6124     | 0.5706     | 0.5843     |
| Score-Level Fusion     | **0.7943** | 0.7846     | 0.7332     | **0.7506** |
| Embedding-Level Fusion | 0.7841     | 0.7610     | 0.7261     | 0.7336     |
| Scalar-Gated Fusion    | 0.7712     | 0.7424     | **0.7399** | 0.7409     |
| Vector-Gated Fusion    | **0.7943** | **0.7966** | 0.7398     | 0.7486     |

All multimodal fusion models outperform the image-only baseline, and the best fusion models (score-level and vector-gated) also surpass the text-only model in accuracy and F1-score.

---

## Hardware

Experiments were conducted using Google Colab with NVIDIA A100 GPU acceleration.

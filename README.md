# Unimodal vs. Multimodal Sentiment Analysis Using Text and Image Data

This project uses the **MVSA-Single** (Multi-View Sentiment Analysis) dataset, which contains social media posts pairing short text captions with images, each labeled with a single sentiment: **positive**, **neutral**, or **negative**. We compare unimodal approaches (text-only, image-only) against multimodal fusion approaches for sentiment analysis. Multimodal models combine both modalities using fusion techniques including **score-level**, **embedding-level**, **scalar-gated**, and **vector-gated** fusion.

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

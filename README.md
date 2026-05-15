# Unimodal vs. Multimodal Sentiment Analysis Using Text and Image Data

## Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

Download the MVSA-Single dataset and place the extracted folder in the project root directory.

Expected structure:

```text
project/
│
├── MVSA_Single/
├── train_text.py
├── train_image.py
├── train_score_fusion.py
├── train_embedding_fusion.py
├── train_scalar_gated.py
├── train_vector_gated.py
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

## Hardware

Experiments were conducted using Google Colab with NVIDIA A100 GPU acceleration.

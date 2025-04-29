# Multi-Task Learning with Transformers: SST-2 & STS-B

This repository contains a PyTorch implementation of a multi-task learning model that jointly trains on the GLUE SST-2 (sentiment classification) and STS-B (semantic textual similarity regression) tasks using a shared transformer encoder. The code demonstrates advanced techniques such as gradual unfreezing, custom sampling, and PCGrad for multi-task optimization.

---

## Features

- **Shared transformer encoder** (e.g., `all-mpnet-base-v2`) with mean pooling
- **Two task-specific heads:**
  - SST-2: Binary classification (sentiment analysis)
  - STS-B: Regression (semantic similarity)
- **Gradual unfreezing** of transformer layers for stable fine-tuning
- **Weighted random sampling** to balance datasets of different sizes
- **PCGrad optimizer** for multi-task gradient conflict resolution
- **Training and evaluation loops** with metrics (accuracy, Pearson correlation)
- **Data loading and batching** with HuggingFace Datasets and Transformers

---

## Requirements

- Python 3.7+
- PyTorch
- Transformers (`sentence-transformers` and `transformers`)
- HuggingFace Datasets
- pytorch_optimizer (for PCGrad)
- NumPy
- SciPy
- Matplotlib

Install dependencies with:

```pip install torch transformers datasets pytorch_optimizer numpy scipy matplotlib```

---

## Usage

### 1. Configure Hyperparameters

Edit the following at the top of the script if desired:

PRETRAINED_MODEL = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 32
NUM_EPOCHS = 4
MAX_LENGTH = 512
LR = 2e-5
WARMUP_STEPS = 100


### 2. Run the Script

The script is self-contained. To train the model:

```python Untitled-1.py```


### 3. Training Details

- The model uses a shared transformer encoder and two output heads.
- **Gradual unfreezing:** The encoder is frozen at first, then the last 3 layers are unfrozen at epoch 2, and the last 6 at epoch 3.
- **PCGrad** is used to resolve conflicting gradients between tasks.
- Training and validation metrics are printed each epoch.

---

## Code Structure

- **MultiTaskModel:** Shared encoder with two heads (classification & regression)
- **Datasets:** Custom PyTorch Dataset wrappers for SST-2 and STS-B from GLUE
- **collate_fn:** Handles tokenization and batching for both tasks
- **DataLoaders:** Includes oversampling for STS-B to match SST-2 size
- **Training Loop:** Alternates between batches from both tasks, computes losses, applies PCGrad, and updates metrics

---

## Evaluation

- **SST-2:** Reports accuracy on train and validation sets
- **STS-B:** Reports Pearson correlation on train and validation sets

---

## Notes

- The script uses CUDA if available.
- The code can be extended to other transformer backbones or tasks by modifying the dataset and model head logic.
- For inference or further evaluation, adapt the final test DataLoader sections.

---

## References

- [GLUE Benchmark](https://gluebenchmark.com/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [pytorch_optimizer (PCGrad)](https://github.com/jettify/pytorch-optimizer)

---

## License

This project is provided for educational purposes. Adapt and modify as needed for your research or projects.
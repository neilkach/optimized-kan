# HPML Project: [Optimizing Kolmogorov-Arnold Networks: Accelerating Training in KANs]

## Team Information
- **Team Name**: [KAN GAN]
- **Members**:
  - Abhishek Chaudhary (ac5003)
  - Neil Biju Kachappilly (nbk2122)
  - Greg Ou (gyo2102)

---

## 1. Problem Statement
Kolmogorov-Arnold Networks (KANs) offer improved interpretability and accuracy compared to standard Multi-Layer Perceptrons (MLPs) by replacing nodal activations with learnable spline-based functions on edges. However, their adoption is limited due to significant computational overhead during training and inference. This project investigates whether modern high-performance machine learning (HPML) techniques—such as mixed precision training, TorchInductor compilation, and quantization-aware training—can be used to accelerate KANs (including FastKAN variants), making them viable alternatives to MLPs in real-world deployments.

---

## 2. Model Description
We work with two variants of Kolmogorov-Arnold Networks:
- Vanilla KAN: Uses B-spline activation functions on edges.
- FastKAN: An architecture that replaces splines with Gaussian radial basis functions (RBFs) to simplify evaluation and improve parallelizability.

Both models were implemented in PyTorch, starting from the [KindXiaoming/pykan]([https://github.com/user/repo/blob/branch/other_file.md](https://github.com/KindXiaoming/pykan) repository. We modified the codebase to support:
- Mixed precision training using torch.cuda.amp
- Torch compilation via torch.compile() with Triton/CUDA graph support and TorchInductor
- Quantization-aware training (QAT) for FastKAN using x86.qconfig
- Custom Black-Scholes dataset generation for benchmarking
 
---

## 3. Final Results Summary

Example Table: 

| Metric               | Value       |
|----------------------|-------------|
| Final Top-1 Accuracy | XX.XX%      |
| Inference Latency    | XX.XX ms    |
| Model Size           | XX MB       |
| Peak Memory Use      | XX MB       |
| Training Time/Epoch  | XX s        |
| Device               | NVIDIA T4 (GCP VM) |

---

## 4. Reproducibility Instructions

### A. Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

---

### B. Log Into WandB or Offline Mode

In order to visualize our results, make sure you have an authenticated WandB account before you run the code. In order to run in offline mode, run the following in shell.
```bash
export WANDB_MODE=offline
python train.py
```

---

### C. Training & Evaluation

To train evaluate the various KAN models, navigate to `black_scholes_mp_example.ipynb`:
```bash
Click "Run All" in notebook.
```

---

### D. Quickstart

To quickly reproduce our resuts, run the following:

```bash
# Step 1: Set up environment
pip install -r requirements.txt

# Step 2: Locate source file
cd black_scholes_mp_example.ipynb

# Step 4: Run training
Click "Run All" in notebook.

```

---

## 5. Notes
- 

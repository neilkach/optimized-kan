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

## 3. Code Repository
The implementation of our several optimizations (including mixed precision and quantization) can be found in the `kan/` directory. Several `.ipynb` files at root were used for testing and visualization. Notably, `mp_example.ipynb` contains the two-variable exponential function initially used to benchmark mixed precision KANs, and `black_scholes_mp_example.ipynb` contains all the code for our final results and visualizations in the final paper, evaluated on the 5-variable Black-Scholes Call Options Pricing Model. A list of notable experiments we ran to test our implementations can be found in the following notebook files.
- `black_scholes_mp_example.ipynb`
- `MP_Wandb.ipynb`
- `Quantization_wandb.ipynb`
- `WandB.ipynb`
- `mp_example.ipynb`
- `plotting.ipynb`
- `quantization.ipynb`

---

## 4. Final Results Summary

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

## 5. Reproducibility Instructions

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

## 6. WandB Project
Our final results and visualizations can be found at the following [WandB Project](https://wandb.ai/hpml_project_spring25/Profiling_Speedups)

## 7. Notes
- The FastKAN implementation on which we applied our optimizations was inspired by and based on the following github repo: [FastKAN](https://github.com/ZiyaoLi/fast-kan)

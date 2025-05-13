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

Both models were implemented in PyTorch, starting from the [KindXiaoming/pykan]([https://github.com/KindXiaoming/pykan](https://github.com/KindXiaoming/pykan) repository. We modified the codebase to support:
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

| Model                                   | Relative CUDA Speed Up (Vanilla KAN / Model) |
|----------------------------------------|----------------------------------------------|
| QuantFastKAN Compiled w/ Options       | 6.27                                         |
| QuantFastKAN Compiled                  | 6.08                                         |
| QuantFastKAN                           | 6.10                                         |
| FastMPKAN Compiled w/ Options          | 7.68                                         |
| FastMPKAN Compiled                     | 7.68                                         |
| FastMPKAN                              | 7.68                                         |
| FastKAN Compiled w/ Options            | 8.01                                         |
| FastKAN Compiled                       | 7.72                                         |
| FastKAN                                | 7.83                                         |
| Mixed Precision KAN Compiled w/ Options| 1.29                                         |
| Mixed Precision KAN Compiled           | 1.29                                         |
| Mixed Precision KAN                    | 1.23                                         |
| Vanilla KAN Compiled w/ Options        | 1.05                                         |
| Vanilla KAN Compiled                   | 1.06                                         |
| **Vanilla KAN**                        | **1.00**                                     |



| Model                         | Ratio to Vanilla KAN (×) |
|-----------------------------------------|---------------------------|
| QuantFastKAN Compiled w/ Options        | 49.93                     |
| QuantFastKAN Compiled                   | 14.20                     |
| QuantFastKAN                            | 31.71                     |
| FastMPKAN Compiled w/ Options           | 31.84                     |
| FastMPKAN Compiled                      | 42.53                     |
| FastMPKAN                               | 45.17                     |
| FastKAN Compiled w/ Options             | 4.67                      |
| FastKAN Compiled                        | 12.31                     |
| FastKAN                                 | 48.08                     |
| Mixed Precision KAN Compiled w/ Options | 7.00                      |
| Mixed Precision KAN Compiled            | 8.07                      |
| Mixed Precision KAN                     | 1.03                      |
| Vanilla KAN Compiled w/ Options         | 4.18                      |
| Vanilla KAN Compiled                    | 3.50                      |
| **Vanilla KAN**                         | **1.00**                  |
| **Input Size**                          | **10,000**                |
| **Device**               | **NVIDIA T4 (GCP VM)**       |
---

| Model                                   | Validation Loss Improvement (Vanilla KAN / Model) |
|----------------------------------------|----------------------------------------------------|
| QuantFastKAN Compiled w/ Options       | 1.10                                               |
| QuantFastKAN Compiled                  | 1.23                                               |
| QuantFastKAN                           | 1.18                                               |
| FastMPKAN Compiled w/ Options          | 1.18                                               |
| FastMPKAN Compiled                     | 1.17                                               |
| FastMPKAN                              | 1.20                                               |
| FastKAN Compiled w/ Options            | 1.15                                               |
| FastKAN Compiled                       | 1.05                                               |
| FastKAN                                | 1.16                                               |
| Mixed Precision KAN Compiled w/ Options| 1.06                                               |
| Mixed Precision KAN Compiled           | 1.07                                               |
| Mixed Precision KAN                    | 1.01                                               |
| Vanilla KAN Compiled w/ Options        | 1.06                                               |
| Vanilla KAN Compiled                   | 1.07                                               |
| **Vanilla KAN**                        | **1.00**                                           |
| **Input Size**                          | **10,000**                |
| **Device**               | **NVIDIA T4 (GCP VM)**       |


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
```
Then follow the E. Quickstart

### C. Training & Evaluation

To train the analogous MLP to KAN, navigate to 
`WandB.ipynb`:
```bash
Click "Run All" in notebook.
```

To train evaluate the various KAN models, navigate to `black_scholes_mp_example.ipynb`:
```bash
Click "Run All" in notebook.
```
Should you want to modify the parameters across the models change
`WandB Specific`, `Dataset Specific`, `Model Specific`, `Training Specific` parameters before initialization 

---

### D. Quickstart

To quickly reproduce our resuts, run the following:

```bash
# Step 1: Set up environment
pip install -r requirements.txt

# Step 2: Locate source file
cd black_scholes_mp_example.ipynb

# Step 3: Verify you are using a GPU VM

# Step 4: Run training
Click "Run All" in notebook.

```

---

## 6. WandB Project
Our final results and visualizations can be found at the following [WandB Project](https://wandb.ai/hpml_project_spring25/Profiling_Speedups)

## 7. Notes
- The FastKAN implementation on which we applied our optimizations was inspired by and based on the following github repo: [FastKAN](https://github.com/ZiyaoLi/fast-kan)
- Model Architecture files are located in `kan/`
- Traces are saved in `traces/`
- Data for visualizations were exported from WandB and moved into `data/`

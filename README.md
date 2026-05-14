# Part 1 – Neural Network Fundamentals and Training Behavior Analysis

> **Course**: Applied Neural Networks, Computer Vision, NLP, and AI Solution Design  
> **Dataset**: Customer Churn Neural Network Dataset (`customer_churn_nn.csv`)  
> **Framework**: TensorFlow 2 / Keras · scikit-learn · Pandas · Matplotlib · Seaborn

---

##  Repository Structure

```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb                         ← All 6 tasks (runnable end-to-end)
├── requirements.txt
├── customer_churn_nn.csv                  ← Dataset
└── results/
    ├── model_comparison_table.csv         ← Hyperparameter results (machine-readable)
    ├── model_comparison_table.png         ← Styled comparison table image
    └── evaluation_outputs.png             ← Full evaluation panel (9 subplots)
```

---

##  Problem Statement

Predict whether a customer will **churn** (leave the service) using structured tabular data.  
The focus is not only on model accuracy but on demonstrating a thorough understanding of how neural networks **learn** — through forward pass, loss calculation, backpropagation, and weight updates.

---

##  Dataset Overview

| Property | Value |
|---|---|
| Rows | 2,000 |
| Columns | 17 (16 features + 1 target) |
| Target | `churn` — 1 = churned, 0 = retained |
| Missing Values | None |
| Class Balance | 98.45% retained / **1.55% churned**  |
| Categorical Features | `region`, `plan_type`, `contract_type`, `payment_method` |
| Numerical Features | tenure, charges, login days, tickets, delays, data usage, satisfaction, complaint recency, discounts, referrals |

>  **Key challenge**: The dataset has a 63:1 class imbalance. Accuracy alone is a misleading metric — F1-Score and ROC-AUC are the primary evaluation criteria.

---

## 🔧 Preprocessing Pipeline

| Step | Action | Detail |
|---|---|---|
| 1 | Drop identifier | Remove `customer_id` (non-predictive) |
| 2 | Encode categoricals | One-Hot Encoding → 28 total input features |
| 3 | Scale numericals | StandardScaler: mean=0, std=1 |
| 4 | Train/Test split | 80/20 stratified split |
| 5 | Class weighting | Churn class weighted 63× to compensate for imbalance |

---

##  Model Architecture (Baseline)

```
Input (28 features)
    ↓
Dense(64, ReLU) → Dropout(0.3)
    ↓
Dense(32, ReLU) → Dropout(0.3)
    ↓
Dense(1, Sigmoid)  →  P(churn)
```

| Component | Choice | Rationale |
|---|---|---|
| Loss | Binary Cross-Entropy | Standard for binary classification |
| Optimizer | Adam (lr = 0.001) | Adaptive, converges reliably |
| Hidden activation | ReLU | Prevents vanishing gradients; sparse representation |
| Output activation | Sigmoid | Maps logit → [0, 1] churn probability |
| Regularisation | Dropout (30%) | Prevents overfitting to the majority class |
| Class Weights | {0: 1.0, 1: 63.0} | Corrects severe 63:1 imbalance |

---

##  Task 4 – Baseline Model Results

| Metric | Value |
|---|---|
| Training Accuracy | 94.41% |
| Validation Accuracy | 97.08% |
| Training Loss | 0.2236 |
| Validation Loss | 0.0836 |
| **Test F1-Score** | **0.18** |
| **Test ROC-AUC** | **0.81** |

**Confusion Matrix (Test Set)**

```
                  Predicted: Retained   Predicted: Churned
Actual: Retained       380                   14
Actual: Churned          4                    2
```

> F1 = 0.18 is low by conventional standards but expected at a 63:1 ratio.  
> ROC-AUC = 0.81 confirms the model meaningfully separates churners from non-churners.

---

##  Task 5 – Hyperparameter Comparison

| Model | Architecture | LR | Batch | Activation | Train Acc | Val Acc | **Test F1** | **ROC-AUC** |
|---|---|---|---|---|---|---|---|---|
| **Baseline** | [64, 32] | 0.001 | 32 | ReLU | 0.9441 | 0.9708 | **0.1818** | 0.8101 |
| Exp 1 – Deeper | [128, 64, 32] | 0.001 | 32 | ReLU | 0.9860 | 0.9750 | 0.1000 | 0.8033 |
| Exp 2 – Shallow | [32] | 0.0001 | 64 | tanh | 0.6368 | 0.6708 | 0.0732 | **0.8790** |
| Exp 3 – Wide | [256, 128] | 0.01 | 16 | ReLU | 0.8985 | 0.9333 | 0.1053 | 0.8041 |

**Key Observations:**

- **Baseline** delivers the best F1 — most practical for real-world churn detection
- **Exp 2** achieves the highest AUC despite underfitting; threshold tuning could unlock its potential
- **Exp 1** overfits the majority class — high training accuracy, low minority-class F1
- **Exp 3** is unstable due to the high learning rate (lr = 0.01) producing noisy loss curves

See `results/model_comparison_table.csv` for full metrics and `results/evaluation_outputs.png` for visual comparison.

---

##  Task 6 – Final Reflection

### Q1 – Role of Weights and Biases

**Weights (W)** scale each input feature at every neuron — a large positive weight amplifies a feature's signal; a negative weight suppresses it. **Biases (b)** shift the activation threshold independently of inputs, giving the model flexibility to fire even when all inputs are zero.

At each neuron: **z = W · x + b**, then **a = f(z)**

During training, backpropagation computes **∂L/∂W** and **∂L/∂b** at every layer. The Adam optimiser uses these gradients to update both towards a lower binary cross-entropy loss — this iterative adjustment is how the network learns which features predict churn.

---

### Q2 – Why Activation Functions are Required

Without non-linear activation functions, stacking multiple Dense layers collapses into a single linear transform: **output = Wₙ · … · W₁ · x + constants**. This cannot model the curved, non-linear decision boundaries that separate churners from non-churners.

- **ReLU** *(f(z) = max(0, z))*: introduces non-linearity, avoids vanishing gradients, enables deep stacking
- **Sigmoid** *(output layer)*: squashes the final value to [0, 1] → directly interpretable as churn probability

---

### Q3 – Learning Rate Effects

| Rate | Effect | Seen In |
|---|---|---|
| **Too High** (0.01) | Gradient steps overshoot the minimum; loss oscillates or diverges | Exp 3 — noisy curves, lower AUC |
| **Too Low** (0.0001) | Convergence is painfully slow; model underfits in the allowed epochs | Exp 2 — train loss 1.15 after 80 epochs |
| **Optimal** (0.001) | Smooth, steady descent to a good minimum | Baseline — best F1, stable curves |

---

### Q4 – Overfitting / Underfitting Analysis

| Model | Diagnosis | Evidence |
|---|---|---|
| **Baseline** | Mild overfitting | Train loss (0.22) > Val loss (0.08) — expected with Dropout active only at train time |
| **Exp 1 (Deeper)** | Stronger overfitting | Train acc 98.6%, but test F1 only 0.10 — memorised majority class |
| **Exp 2 (Shallow/Low LR)** | Clear underfitting | Train acc 63.7%, train loss 1.15 — model never converged |
| **Exp 3 (Wide/High LR)** | Training instability | Erratic loss, oscillating accuracy — step size too large |

**Root cause throughout**: The 63:1 class imbalance dominates all training dynamics. Even a naïve "always predict retained" model achieves 98.45% accuracy. Future improvements should include SMOTE oversampling, Focal Loss, or threshold optimisation on the precision-recall curve.

---

##  How to Run

```bash
# 1. Clone repo
git clone https://github.com/<your-username>/part-1-neural-network-analysis
cd part-1-neural-network-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook notebook.ipynb
```

All outputs are saved automatically to `results/`.

---

##  Requirements

```
tensorflow>=2.10
scikit-learn>=1.2
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
jupyter>=1.0
```
# 🔧 Models From Scratch

Pure NumPy implementations of machine learning models, built to mirror the scikit-learn API. Each model is implemented from first principles — covering the forward pass, analytical gradients, multiple loss functions, regularization, and adaptive learning rate schedules.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

---

## 📂 Models Overview

| Model | Type | Loss Functions | Regularization | Learning Rate Schedules |
| :--- | :--- | :--- | :--- | :--- |
| **SGDClassifier** | Binary Classification | Hinge, Squared Hinge, Log Loss, Modified Huber, Perceptron | L1, L2, ElasticNet | constant, optimal, invscaling, adaptive |
| **SGDRegressor** | Regression | Squared Error, Huber, ε-Insensitive, Squared ε-Insensitive | L1, L2, ElasticNet | constant, optimal, invscaling, adaptive |

---

## 📐 SGDClassifier

A binary classifier trained with stochastic gradient descent, implementing the same interface and options as `sklearn.linear_model.SGDClassifier`.

**Tested on:** `make_classification` (1,000 samples, 4 informative features) — achieved **93.3% accuracy**.

### Loss Functions

| Loss | Formula | Use Case |
| :--- | :--- | :--- |
| `hinge` | `max(0, 1 − y·ŷ)` | Linear SVM |
| `squared_hinge` | `max(0, 1 − y·ŷ)²` | Smooth SVM margin |
| `log_loss` | `log(1 + e^(−y·ŷ))` | Logistic regression |
| `modified_huber` | `max(0, 1−y·ŷ)²` if `y·ŷ ≥ −1` else `−4y·ŷ` | Outlier-robust logistic |
| `perceptron` | `max(0, −y·ŷ)` | Perceptron rule |

### Implementation Details

- Weights initialized from `N(0, 0.5)`; bias initialized to 0
- Mini-batch sampling without replacement per epoch (`sample_size` parameter, default 30%)
- Analytical gradients computed for both weights (`∂L/∂W`) and bias (`∂L/∂b`) for all loss functions
- Intercept update controlled by `fit_intercept` flag
- Convergence check: training stops when `|loss_t − loss_{t-1}| < tol`

---

## 📈 SGDRegressor

A regression model trained with stochastic gradient descent, supporting SVR-style losses in addition to the standard squared error.

**Tested on:** A synthetic nonlinear dataset (`3·sin(x) + (0.1x)²`) — converged to a final loss of **~0.0084**.

### Loss Functions

| Loss | Description | Use Case |
| :--- | :--- | :--- |
| `squared_error` | Mean squared residuals | Standard linear regression |
| `huber` | MSE for small residuals, MAE for large ones (threshold: `ε`) | Outlier-robust regression |
| `epsilon_insensitive` | Zero loss inside the `ε`-tube (SVR) | Support Vector Regression |
| `squared_epsilon_insensitive` | Squared penalty outside the `ε`-tube | Smooth SVR |

### Implementation Details

- Huber and ε-insensitive losses require separate gradient paths for residuals inside and outside the threshold region — both are implemented analytically
- Residual vector `ŷ − y` reused across forward and backward passes to avoid redundant computation
- Same mini-batch SGD loop, learning rate schedules, and regularization as the classifier

---

## ⚙️ Shared Architecture

Both models share a common design:

**Learning Rate Schedules**

| Schedule | Update Rule |
| :--- | :--- |
| `constant` | `η = η₀` |
| `optimal` | `η = 1 / (α · (t + 1/(α·η₀)))` |
| `invscaling` | `η = η₀ / (t+1)^power_t` |
| `adaptive` | Divide `η` by 5 when loss increases |

**Regularization**

| Penalty | Loss Term |
| :--- | :--- |
| `l1` | `α · ‖W‖₁` |
| `l2` | `α · ‖W‖₂²` |
| `elasticnet` | `α · ((1−ρ)·‖W‖₂² + ρ·‖W‖₁)` |

**API** — both models follow the scikit-learn convention:

```python
model = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, learning_rate='optimal')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 📁 Structure

```
models-from-scratch/
├── classification/
│   ├── sgd_classifier.py       # SGDClassifier implementation
│   └──  losses.py              # Classification loss functions & gradients
└── regression/
    ├── sgd_regressor.py        # SGDRegressor implementation
    └── losses.py               # Regression loss functions & gradients
```

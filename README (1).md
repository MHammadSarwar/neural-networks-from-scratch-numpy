
# Neural Network from Scratch

## 📌 Project Overview

This notebook builds a neural network from first principles using NumPy. It emphasizes **mathematical clarity**, **manual computation of steps**, and **educational transparency**, focusing on **binary classification** using a simple 2-layer feedforward neural network.

---

## 🧠 Network Architecture

- **Input**: 2 features  
- **Hidden Layer**: 3 neurons with activation  
- **Output Layer**: 1 neuron (sigmoid)  
- **Loss Function**: Binary Cross Entropy  
- **Optimization**: Manual gradient descent update  

```
Input (2D) → Dense (3N) → Activation → Dense (1N) → Sigmoid → Loss
```

---

## 📊 Dataset

- **Type**: Synthetic  
- **Shape**:
  - Input features `X`: (5, 2)
  - Labels `y`: (5,)
- **Interpretation**:  
  - Feature 1 = IQ  
  - Feature 2 = GPA  
  - Label = Intelligence (0 or 1)

```python
X = np.array([[0.5, 1.5],
              [1.0, 2.0],
              [1.5, 0.5],
              [2.0, 3.0],
              [3.0, 1.0]])
y = np.array([0, 0, 1, 1, 1])
```

---

## 🔧 Implementation Steps

### Step 1: Data Generation  
- Small dataset to allow **manual tracing of forward and backward propagation**.  
- Matrix format used for efficient computation.

### Step 2: Parameter Initialization  
- Weight Matrices:  
  - `W1`: shape (2, 3)  
  - `W2`: shape (3, 1)  
- Bias Vectors:  
  - `b1`: shape (1, 3)  
  - `b2`: shape (1, 1)  
- Initialization using small random values for symmetry breaking.

### Step 3: Forward Propagation  
- Layer 1:  
  \( Z1 = X \cdot W1 + b1 \)  
  \( A1 = 	ext{ReLU}(Z1) \)  
- Output Layer:  
  \( Z2 = A1 \cdot W2 + b2 \)  
  \( \hat{y} = \sigma(Z2) \)

### Step 4: Loss Computation  
- Binary Cross-Entropy Loss:  
  \[
  L = -\frac{1}{m} \sum \left[ y \log(\hat{y}) + (1 - y)\log(1 - \hat{y}) \right]
  \]

### Step 5: Backward Propagation  
- Manual derivation of gradients:  
  - dW2, db2  
  - dW1, db1  
- Chain rule applied across layers.  
- Learning rate used for updates.

### Step 6: Parameter Update  
- Gradient descent with static learning rate.

---

## 🧪 Training

- Iterative training loop implemented.
- Print statements for intermediate loss values.
- No external ML library used—**pure NumPy implementation**.

---

## 📈 Outputs and Results

- Accuracy checked manually.
- Loss convergence visible.
- Interpretation discussed in context of data (IQ, GPA, intelligence).

---

## 📚 Key Concepts Demonstrated

- Matrix-based neural network operations  
- Forward and backward propagation  
- Chain rule and gradient derivation  
- Manual weight updates  
- Shape validation throughout

---

## 📦 Requirements

```bash
python3
numpy
matplotlib (if visualization is extended)
```

---

## ✅ Usage

To run the notebook:

```bash
jupyter notebook N_E_U_R_A_L_N_E_T_W_O_R_K.ipynb
```

Modify learning rates or number of neurons in hidden layers to explore learning behavior.

---

## ✍️ Author Notes

This notebook is intended for **educational** purposes. It provides a clear mathematical walk-through of neural networks for students and engineers new to deep learning theory.

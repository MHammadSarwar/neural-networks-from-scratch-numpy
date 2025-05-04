# 🧠 Neural Network from Scratch

## 📌 Project Overview
This project demonstrates how to build a neural network **from first principles using NumPy**. The goal is to provide educational clarity by manually implementing each computation step. It focuses on a **binary classification task** using a simple **2-layer feedforward neural network**.

## 🧠 Network Architecture
- **Input Layer:** 2 features  
- **Hidden Layer:** 3 neurons with activation  
- **Output Layer:** 1 neuron with sigmoid activation  
- **Loss Function:** Binary Cross Entropy  
- **Optimization:** Manual Gradient Descent

**Flow Diagram:**  
`Input (2D) → Dense (3N) → Activation → Dense (1N) → Sigmoid → Loss`

## 📊 Dataset
- **Type:** Synthetic  
- **Input Features (X):** Shape `(5, 2)`  
- **Labels (y):** Shape `(5,)`

### Feature Interpretation
- **Feature 1:** IQ  
- **Feature 2:** GPA  
- **Label:** Intelligence (`0` = Not Intelligent, `1` = Intelligent)
## 🔧 Implementation Steps

### Step 1: Data Generation
- A small dataset is used to allow **manual tracing** of forward and backward propagation.
- Data is represented in **matrix format** for efficient computation.

### Step 2: Parameter Initialization
- **Weight Matrices:**
  - `W1`: shape `(2, 3)`
  - `W2`: shape `(3, 1)`
- **Bias Vectors:**
  - `b1`: shape `(1, 3)`
  - `b2`: shape `(1, 1)`
- Parameters are initialized with **small random values** to break symmetry.

### Step 3: Forward Propagation
- **Layer 1:**
  - \( Z_1 = X \cdot W_1 + b_1 \)
  - \( A_1 = \text{ReLU}(Z_1) \)
- **Output Layer:**
  - \( Z_2 = A_1 \cdot W_2 + b_2 \)
  - \( \hat{y} = \sigma(Z_2) \)

### Step 4: Loss Computation
- **Binary Cross-Entropy Loss:**
  \[
  L = -\frac{1}{m} \sum \left[ y \log(\hat{y}) + (1 - y)\log(1 - \hat{y}) \right]
  \]

### Step 5: Backward Propagation
- Manual derivation of gradients:
  - `dW2`, `db2`
  - `dW1`, `db1`
- **Chain rule** is applied across layers.
- A static **learning rate** is used for updates.

### Step 6: Parameter Update
- Parameters are updated using **gradient descent** with a **fixed learning rate**.
## 🧪 Training
- An **iterative training loop** is implemented.
- Includes **print statements** to monitor intermediate loss values.
- No external ML libraries are used — implementation is **pure NumPy**.

## 📈 Outputs and Results
- **Accuracy** is checked manually.
- **Loss convergence** is visually and numerically evident.
- Results are interpreted in the context of the dataset (IQ, GPA, intelligence).

## 📚 Key Concepts Demonstrated
- Matrix-based neural network operations  
- Forward and backward propagation  
- Chain rule and gradient derivation  
- Manual weight updates  
- Shape validation at each step  

## 📦 Requirements
- `python3`  
- `numpy`  
- `matplotlib` *(optional, for extended visualizations)*

## ✅ Usage
To run the notebook:

```bash
jupyter notebook N_E_U_R_A_L_N_E_T_W_O_R_K.ipynb
## ✍️ Author Notes

This notebook is created for **educational purposes**. It provides a **clear, step-by-step mathematical walk-through** of how neural networks work, aimed at **students and engineers new to deep learning**.

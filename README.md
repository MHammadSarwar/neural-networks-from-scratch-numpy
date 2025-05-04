# Neural Network from Scratch

## 📌 Project Overview  
This notebook builds a neural network from first principles using NumPy. It emphasizes **mathematical clarity**, **manual computation of steps**, and **educational transparency**, focusing on **binary classification** using a simple 2-layer feedforward neural network.

---

## 🧠 Network Architecture  
- **Input**: 2 features  
- **Hidden Layer**: 3 neurons with ReLU activation  
- **Output Layer**: 1 neuron (sigmoid)  
- **Loss Function**: Binary Cross Entropy  
- **Optimization**: Manual gradient descent  
Input (2D) → Dense (3N) → ReLU → Dense (1N) → Sigmoid → Loss


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
🔧 Implementation Steps
1. Data Generation
5 synthetic samples for manual forward/backward pass tracing

Matrix format for batch processing

2. Parameter Initialization
python
W1.shape = (2, 3), W2.shape = (3, 1)  
b1.shape = (1, 3), b2.shape = (1, 1)  
3. Forward Propagation
Z1 = X·W1 + b1 → ReLU  
Z2 = A1·W2 + b2 → Sigmoid  
4. Loss Computation
Binary Cross-Entropy: L = -1/m Σ[y log(ŷ) + (1-y)log(1-ŷ)]
5. Backward Propagation
Manual chain rule implementation

Gradient calculations for:

dW2, db2 (output layer)

dW1, db1 (hidden layer)

6. Parameter Update
W = W - α·dW  
b = b - α·db  
🧪 Training
Pure NumPy implementation (no ML libraries)

Training loop with loss tracking

Configurable epochs/learning rate

python
train_model(X, y, epochs=1000, learning_rate=0.01)  
📈 Results
Prediction probabilities & accuracy scoring

Confusion matrix analysis

Loss convergence visualization

📚 Key Concepts Demonstrated
Matrix operations for neural networks

Forward/backward propagation mechanics

Gradient descent optimization

Shape consistency validation

📦 Requirements
bash
Python 3  
NumPy  
Matplotlib (optional for visualization)  
✅ Usage
Clone repository

Run Jupyter notebook:

bash
jupyter notebook NEURAL_NETWORK.ipynb  
Modify hyperparameters:

python
# Example  
train_model(X, y, epochs=2000, learning_rate=0.1)  
✍️ Author Notes
Built for educational purposes - demonstrates core neural network math without abstraction. Ideal for learners wanting to understand DL fundamentals.


This version:
1. Uses GitHub-friendly markdown syntax
2. Maintains clear section hierarchy
3. Preserves code blocks and mathematical notation
4. Optimizes spacing for readability
5. Adds emoji visual hierarchy
6. Keeps all implementation details intact

You can directly copy-paste this into a `README.md` file in your GitHub repository.

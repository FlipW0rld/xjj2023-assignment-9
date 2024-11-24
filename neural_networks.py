import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
from functools import partial  # Add this line
import matplotlib
matplotlib.use('Agg')


result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        self.activations = {
            'tanh': (np.tanh, lambda z: 1 - np.tanh(z)**2),
            'relu': (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
            'sigmoid': (lambda z: 1 / (1 + np.exp(-z)), lambda z: z * (1 - z))
        }
        self.activation, self.activation_derivative = self.activations[self.activation_fn]

    def forward(self, X):
        self.X = X
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activation(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.tanh(self.Z2)  # Output activation (change if necessary)
        return self.A2

    def backward(self, X, y):
        m = X.shape[0]
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, self.W2.T) * self.activation_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Gradient descent step
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def get_gradients(self):
        return np.abs(self.W1), np.abs(self.W2)

# Generate synthetic dataset
def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization function
def update(frame, mlp, ax_hidden, ax_input, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Training step
    for _ in range(10):  # Accumulate 10 steps for smoother animation
        mlp.forward(X)
        mlp.backward(X, y)

    # Hidden layer visualization
    hidden_features = mlp.A1
    ax_hidden.scatter(
        hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7
    )
    ax_hidden.set_title(f"Hidden Space at Step {frame}")
    ax_hidden.set_xlim(-1, 1)
    ax_hidden.set_ylim(-1, 1)
    ax_hidden.set_zlim(-1, 1)

    # Input space decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, alpha=0.8, cmap='bwr')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolor='k', cmap='bwr')
    ax_input.set_title(f"Input Space at Step {frame}")

    # Gradient visualization
    edge_thickness_W1, edge_thickness_W2 = mlp.get_gradients()
    node_positions = {"x1": (0, 1), "x2": (0, 0), "h1": (0.5, 0.8), "h2": (0.5, 0.5), "h3": (0.5, 0.2), "y": (1, 0.5)}
    edges = [("x1", "h1"), ("x1", "h2"), ("x1", "h3"),
             ("x2", "h1"), ("x2", "h2"), ("x2", "h3"),
             ("h1", "y"), ("h2", "y"), ("h3", "y")]
    
    for edge, thickness in zip(edges, np.concatenate([edge_thickness_W1.flatten(), edge_thickness_W2.flatten()])):
        x1, y1 = node_positions[edge[0]]
        x2, y2 = node_positions[edge[1]]
        ax_gradient.plot([x1, x2], [y1, y2], "b-", linewidth=thickness * 5, alpha=0.6)

    for node, pos in node_positions.items():
        ax_gradient.scatter(*pos, s=1000, c="blue")
        ax_gradient.text(pos[0], pos[1], node, fontsize=12, ha="center", color="white")
    ax_gradient.set_title(f"Gradients at Step {frame}")
    ax_gradient.axis("off")

# Main visualization function
def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_hidden=ax_hidden, ax_input=ax_input, ax_gradient=ax_gradient, X=X, y=y), frames=step_num // 10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

# Run visualization
if __name__ == "__main__":
    visualize(activation="tanh", lr=0.1, step_num=1000)

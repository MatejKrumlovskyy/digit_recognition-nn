"""
Digit Recognition — Noise Robustness Analysis
==============================================
Trains a feedforward NN (64 hidden neurons, ReLU) on binary digit data (0–9).
Tests how classification accuracy degrades as more bits are replaced
with random values (noise injection).

X-axis: number of noisy bits (0 to 10)
Y-axis: accuracy on 100 random noisy variants of digit 0

Author: Matej Krumlovsky
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dataset

inputs = np.array([
    [0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,0,1,1,0], # 0
    [0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0], # 1
    [0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,1,1,1], # 2
    [0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,1,0,1,1,0], # 3
    [0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,1,1,1,0,0,1,0,0,0,1,0], # 4
    [1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,0,1,0,0,0,1,1,1,1,0], # 5
    [0,1,1,0,1,0,0,0,1,0,0,0,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0], # 6
    [1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0], # 7
    [0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0], # 8
    [0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,0,1,1,0]  # 9
])
outputs = np.eye(10)

# Model

model = Sequential([
    Dense(64, input_dim=28, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(inputs, outputs, epochs=50, batch_size=1, verbose=0)
print("Training complete.")

# Noise Injection Helper

def add_noise(vector, num_bits):
    """Replace `num_bits` random positions with random values (2–9)."""
    v = np.copy(vector)
    for idx in np.random.choice(len(v), num_bits, replace=False):
        v[idx] = np.random.randint(2, 10)
    return v

# Evaluation

MAX_NOISY_BITS = 10
NUM_VARIANTS   = 100
TEST_DIGIT     = 0   

accuracies = []

for n in range(0, MAX_NOISY_BITS + 1):
    correct = sum(
        np.argmax(model.predict(np.array([add_noise(inputs[TEST_DIGIT], n)]), verbose=0)) == TEST_DIGIT
        for _ in range(NUM_VARIANTS)
    )
    acc = (correct / NUM_VARIANTS) * 100
    accuracies.append(acc)
    print(f"Noisy bits: {n:2d} | Accuracy: {acc:.1f}%")

# Plot

plt.figure(figsize=(8, 5))
plt.plot(range(0, MAX_NOISY_BITS + 1), accuracies, marker='o', color='steelblue', linewidth=2)
plt.title(f'Accuracy vs. Number of Noisy Bits (digit {TEST_DIGIT})')
plt.xlabel('Number of noisy bits')
plt.ylabel('Accuracy (%)')
plt.xticks(range(0, MAX_NOISY_BITS + 1))
plt.ylim(0, 105)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('results/noise_robustness.png', dpi=150)
plt.show()

# -*- coding: utf-8 -*-
"""
Digit Recognition — Bit-Flip Corruption Analysis
=================================================
Trains a feedforward NN (64 hidden neurons, ReLU) on binary digit data (0–9).
Tests how classification accuracy degrades as more bits are flipped (0↔1).

X-axis: number of flipped bits (1 to 14)
Y-axis: accuracy on 100 random corrupted variants of digit 5

Author: Matej Krumlovsky
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ── Dataset ───────────────────────────────────────────────────────────────────

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

# ── Model ─────────────────────────────────────────────────────────────────────

model = Sequential([
    Dense(64, input_dim=28, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(inputs, outputs, epochs=50, batch_size=1, verbose=0)
print("Training complete.")

# ── Bit-Flip Helper ───────────────────────────────────────────────────────────

def flip_bits(vector, num_flips):
    """Flip `num_flips` random bits (0↔1)."""
    v = np.copy(vector)
    for idx in np.random.choice(len(v), num_flips, replace=False):
        v[idx] = 1 - v[idx]
    return v

# ── Evaluation ────────────────────────────────────────────────────────────────

MAX_FLIPS    = 14
NUM_VARIANTS = 100
TEST_DIGIT   = 5   # digit used for robustness testing

flip_counts = range(1, MAX_FLIPS + 1)
accuracies  = []

for n in flip_counts:
    correct = sum(
        np.argmax(model.predict(np.array([flip_bits(inputs[TEST_DIGIT], n)]), verbose=0)) == TEST_DIGIT
        for _ in range(NUM_VARIANTS)
    )
    acc = (correct / NUM_VARIANTS) * 100
    accuracies.append(acc)
    print(f"Flipped bits: {n:2d} | Accuracy: {acc:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 5))
plt.plot(flip_counts, accuracies, marker='o', color='tomato', linewidth=2)
plt.title(f'Accuracy vs. Number of Flipped Bits (digit {TEST_DIGIT})')
plt.xlabel('Number of flipped bits')
plt.ylabel('Accuracy (%)')
plt.xticks(flip_counts)
plt.ylim(0, 105)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('results/bitflip_robustness.png', dpi=150)
plt.show()

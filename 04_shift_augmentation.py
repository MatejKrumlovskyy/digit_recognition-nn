"""
Digit Recognition — Shift Augmentation & Spatial Invariance
============================================================
Trains a feedforward NN on binary digit data (0–9) augmented with
4-directional shifts (up, down, left, right) of the 7×4 grid.

Tests whether the model correctly classifies shifted variants of each digit,
demonstrating spatial data augmentation for improved invariance.

Author: Matej Krumlovsky
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dataset

original_inputs = np.array([
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
original_outputs = np.eye(10)

# Shift Helper

DIRECTIONS = ['up', 'down', 'left', 'right']

def shift_vector(vector, direction):
    """Shift a 7×4 binary grid one step in the given direction."""
    grid = vector.reshape(7, 4)
    shifted = np.zeros_like(grid)
    if direction == 'up':
        shifted[:-1] = grid[1:]
    elif direction == 'down':
        shifted[1:] = grid[:-1]
    elif direction == 'left':
        shifted[:, :-1] = grid[:, 1:]
    elif direction == 'right':
        shifted[:, 1:] = grid[:, :-1]
    return shifted.flatten()

def visualize_digit(vector, title="Digit"):
    """Display a 7×4 binary grid as an image."""
    plt.figure(figsize=(2, 3))
    plt.imshow(vector.reshape(7, 4), cmap='binary', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Data Augmentation

aug_inputs, aug_outputs = [], []
for i, vec in enumerate(original_inputs):
    aug_inputs.append(vec)
    aug_outputs.append(original_outputs[i])
    for d in DIRECTIONS:
        aug_inputs.append(shift_vector(vec, d))
        aug_outputs.append(original_outputs[i])

aug_inputs  = np.array(aug_inputs)
aug_outputs = np.array(aug_outputs)

print(f"Original samples: {len(original_inputs)} → Augmented: {len(aug_inputs)}")

# Model

model = Sequential([
    Dense(64, input_dim=28, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(aug_inputs, aug_outputs, epochs=11, batch_size=1, verbose=0)
print("Training complete.")

# Evaluation: Per-Direction Accuracy

TEST_DIGIT = 1  

print(f"\nTesting shifted variants of digit {TEST_DIGIT}:")
visualize_digit(original_inputs[TEST_DIGIT], title=f"Original digit {TEST_DIGIT}")

direction_accuracy = {}
for d in DIRECTIONS:
    shifted = shift_vector(original_inputs[TEST_DIGIT], d)
    visualize_digit(shifted, title=f"Digit {TEST_DIGIT} — shifted {d}")

    pred = np.argmax(model.predict(np.array([shifted]), verbose=0))
    correct = int(pred == TEST_DIGIT)
    direction_accuracy[d] = correct * 100
    status = "✓ Correct" if correct else f"✗ Predicted {pred}"
    print(f"  Shift '{d}': {status}")

# Summary

print(f"\nPer-direction accuracy for digit {TEST_DIGIT}:")
for d, acc in direction_accuracy.items():
    print(f"  {d:6s}: {acc:.0f}%")

# Bar Chart

plt.figure(figsize=(6, 4))
plt.bar(direction_accuracy.keys(), direction_accuracy.values(), color='mediumpurple')
plt.title(f'Shift Classification Accuracy — Digit {TEST_DIGIT}')
plt.xlabel('Shift direction')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig('results/shift_augmentation.png', dpi=150)
plt.show()

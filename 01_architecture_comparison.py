"""
Digit Recognition — Architecture & Robustness Comparison
=========================================================
Trains feedforward neural networks on a custom binary digit dataset (0–9)
represented as 7×4 binary grids (28-bit vectors).

For each combination of architecture and activation function, tests:
  - Accuracy on clean data
  - Accuracy on noisy data  (random values injected into bits)
  - Accuracy on corrupted data (bit flipping: 0↔1)

Architectures tested:
  (32,) | (64,) | (128,) | (64, 32) | (128, 64)

Activation functions:
  ReLU | Sigmoid | Tanh

Author: Matej Krumlovsky
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Dataset
# Each digit 0–9 encoded as a 7×4 binary grid (28 bits), read row by row

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

# One-hot encoded labels
outputs = np.eye(10)

# Augmentation Helpers

def add_noise(vector, num_bits):
    """Replace `num_bits` random positions with random values (2–9)."""
    v = np.copy(vector)
    for idx in np.random.choice(len(v), num_bits, replace=False):
        v[idx] = np.random.randint(2, 10)
    return v

def flip_bits(vector, num_flips):
    """Flip `num_flips` random bits (0↔1)."""
    v = np.copy(vector)
    for idx in np.random.choice(len(v), num_flips, replace=False):
        v[idx] = 1 - v[idx]
    return v

# Experiment Configuration

architectures = [
    (32,),
    (64,),
    (128,),
    (64, 32),
    (128, 64),
]

activation_functions = ['relu', 'sigmoid', 'tanh']

NUM_VARIANTS   = 100  
NUM_NOISY_BITS = 5    
NUM_FLIP_BITS  = 5    

# Training & Evaluation

results = []

for config in architectures:
    for activation in activation_functions:
        print(f"Training: arch={config}, activation={activation}")

        model = Sequential()
        model.add(Dense(config[0], input_dim=28, activation=activation))
        for neurons in config[1:]:
            model.add(Dense(neurons, activation=activation))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        model.fit(inputs, outputs, epochs=50, batch_size=1, verbose=0)

        clean_correct = sum(
            np.argmax(model.predict(np.array([x]), verbose=0)) == np.argmax(y)
            for x, y in zip(inputs, outputs)
        )
        clean_acc = (clean_correct / len(inputs)) * 100

        noisy_correct = sum(
            np.argmax(model.predict(np.array([add_noise(inputs[0], NUM_NOISY_BITS)]), verbose=0)) == 0
            for _ in range(NUM_VARIANTS)
        )
        noisy_acc = (noisy_correct / NUM_VARIANTS) * 100
        flip_correct = sum(
            np.argmax(model.predict(np.array([flip_bits(inputs[0], NUM_FLIP_BITS)]), verbose=0)) == 0
            for _ in range(NUM_VARIANTS)
        )
        flip_acc = (flip_correct / NUM_VARIANTS) * 100

        results.append({
            'config': config,
            'activation': activation,
            'clean_acc': clean_acc,
            'noisy_acc': noisy_acc,
            'flip_acc': flip_acc
        })

# Results Summary

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
for r in results:
    print(f"Arch: {str(r['config']):<10} | Act: {r['activation']:<8} | "
          f"Clean: {r['clean_acc']:6.2f}% | "
          f"Noisy: {r['noisy_acc']:6.2f}% | "
          f"Flipped: {r['flip_acc']:6.2f}%")

# Visualization

fig, ax = plt.subplots(figsize=(13, 10))
labels = [f"{r['config']}, {r['activation']}" for r in results]
x = np.arange(len(labels))
w = 0.25

ax.barh(x - w, [r['clean_acc'] for r in results],  w, label='Clean data',   color='skyblue')
ax.barh(x,     [r['noisy_acc'] for r in results],   w, label='Noisy data',   color='salmon')
ax.barh(x + w, [r['flip_acc'] for r in results],    w, label='Flipped data', color='lightgreen')

ax.set_xlabel('Accuracy (%)')
ax.set_yticks(x)
ax.set_yticklabels(labels)
ax.set_title('Neural Network Architecture Comparison — Clean vs Noisy vs Flipped Data')
ax.legend()
plt.tight_layout()
plt.savefig('results/architecture_comparison.png', dpi=150)
plt.show()

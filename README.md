Digit Recognition — Neural Network Robustness Analysis

A feedforward neural network trained to recognize handwritten digits (0–9) encoded as **7×4 binary grids** (28-bit vectors). The project goes beyond basic classification — it systematically analyzes **how and why models fail** under real-world conditions.

---

Project Overview

| Property | Details |
|---|---|
| Task | Digit classification (0–9) |
| Input | Binary 7×4 pixel grid (28 bits) |
| Framework | TensorFlow / Keras |
| Focus | Robustness testing & architecture comparison |

---

Scripts

| File | Description |
|---|---|
| `01_architecture_comparison.py` | Compares 5 architectures × 3 activation functions on clean, noisy and flipped data |
| `02_noise_robustness.py` | Accuracy vs. number of noisy bits (0–10) |
| `03_bitflip_robustness.py` | Accuracy vs. number of flipped bits (1–14) |
| `04_shift_augmentation.py` | Trains on shift-augmented data, tests spatial invariance |

---

## 🧠 What Makes This Project Unique

Unlike standard digit recognition, this project experimentally tests **model robustness**:

- 🔊 **Noise injection** — random values replacing bits (simulates sensor noise)
- 💥 **Bit flipping** — 0↔1 corruption at varying rates
- ↕️ **Input shifting** — up/down/left/right spatial shifts with data augmentation
- 🏗️ **Architecture sweep** — 5 layer configs × 3 activation functions (ReLU, Sigmoid, Tanh)

---

## 🔧 Architectures Tested

```
(32,)      — 1 hidden layer, 32 neurons
(64,)      — 1 hidden layer, 64 neurons
(128,)     — 1 hidden layer, 128 neurons
(64, 32)   — 2 hidden layers
(128, 64)  — 2 hidden layers
```

Each tested with: **ReLU**, **Sigmoid**, **Tanh**

---

## 📊 Key Findings

- Models degrade gracefully with noise — accuracy drops with more corrupted bits
- Bit flipping is more destructive than noise injection at the same count
- Shift augmentation improves spatial invariance significantly
- Deeper networks with ReLU generally outperform shallow ones on noisy data

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt

# Run experiments individually:
python 01_architecture_comparison.py
python 02_noise_robustness.py
python 03_bitflip_robustness.py
python 04_shift_augmentation.py
```

Results (graphs) are automatically saved to the `results/` folder.

---

## 👤 Author

**Matej Krumlovský** — FEI STU Bratislava  
[matejkrumlovsky8@gmail.com](mailto:matejkrumlovsky8@gmail.com)

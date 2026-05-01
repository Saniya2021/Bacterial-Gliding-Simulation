# 🦠 Fjohnsoniae 3D Gliding Simulation

This repository contains a Python simulation project that models the surface 
gliding motility of *Flavobacterium johnsoniae*. The simulation is driven by 
adhesion dynamics where surface units move along helical tracks, stochastically 
binding and unbinding. Direction changes arise when adhesion conditions trigger 
either smooth turns or sharp flips. The code logs trajectories, visualizes paths 
in 2D/3D, and reports metrics such as total turns and turns per unit length.

---

## 📈 Updates

**May 2026** — Upgraded from 2D to full 3D simulation. Corrected drag coefficients 
(Tirado & de la Torre), roll torque cross-product, binding/unbinding logic, and 
trajectory calculations.

---

## 📂 Project Contents

| File | Description |
|------|-------------|
| `Fjohnsoniae-3D-Gliding-Simulation.ipynb` | Current 3D simulation with corrected physics |
| `2DGlidingLogic.py` | Earlier 2D version of the gliding logic |
| `originalGlidingLogic.py` | Original reference implementation |

---

## 🎯 Project Overview

This project demonstrates the following computational and simulation concepts:

- 🧪 Stochastic modeling of motion where binding events influence direction changes
- 🌀 3D trajectory simulation using corrected biophysical parameters
- 📊 Visualization of motion paths in 2D and 3D
- 📈 Logging and analysis of simulation metrics such as turns and movement efficiency

---

## 🛠️ Tech Stack

- **Language:** Python
- **Numerical Computing:** NumPy
- **Visualization:** Matplotlib
- **Development Environment:** Jupyter Notebook

---

## 🛠️ How to Use

1. **Open the notebook**
   Open `Fjohnsoniae-3D-Gliding-Simulation.ipynb` in Jupyter to explore the 
   full 3D simulation with corrected physics and visualizations.

2. **Or run the 2D script**
```bash
   python 2DGlidingLogic.py
```

3. **Requirements**
```bash
   pip install numpy matplotlib
```

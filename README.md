# 🦠 Bacterial Gliding Simulation

This repository contains a Python simulation project that models the motion of bacteria as they glide along surfaces. The simulation is driven by adhesion dynamics where surface units move along helical tracks, stochastically binding and unbinding. Direction changes arise when adhesion conditions trigger either smooth turns or sharp flips. The code logs trajectories, visualizes paths in 2D/3D, and reports metrics such as total turns and turns per unit length.

---

## 📂 Project Contents

The repository includes the following files:

- **2DGlidingLogic.py** – Python script containing logic for simulating bacterial gliding in 2D.  
- **originalGlidingLogic.py** – An original version of the gliding logic for reference or comparison.  
- **README.md** – Project documentation.  

---

## 🎯 Project Overview

This project demonstrates the following computational and simulation concepts:

- 🧪 Stochastic modeling of motion where binding events influence direction changes  
- 🌀 Implementation of simulated trajectories using Python  
- 📊 Visualization of motion paths in two or three dimensions  
- 📈 Logging and analysis of simulation metrics such as turns and movement efficiency  

This project provides a hands-on example of how simulation can be used to model dynamic biological processes.

---

## 🛠️ How to Use

1. **Open the code files**  
   Inspect `2DGlidingLogic.py` or `originalGlidingLogic.py` to understand model logic, parameters, and how motion is simulated.

2. **Run the simulation**  
   In a Python environment that supports the required libraries, run the appropriate script:
   ```bash
   python 2DGlidingLogic.py

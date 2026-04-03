# 🧠 BioSim: Intervention-Aware System Predictor

## 🚀 What This Does
BioSim predicts how a system evolves over time — and how interventions change its future behavior.

**Input:**
- Current system state
- Intervention (external change applied)

**Output:**
- Future trajectory at t+1, t+2, and t+3

BioSim is a computational framework for learning the temporal evolution of biological systems in response to interventions. It combines dynamic systems modeling with a neural network-based predictor to generate multivariate future trajectories.

---

## 💡 Why This Matters
Most machine learning models predict static outputs.

BioSim models **dynamic systems**, where:

state(t) → state(t+1), state(t+2), state(t+3)

This enables simulation of **what-if scenarios**, for example:
- What happens with treatment A versus treatment B?
- How does the system respond over time?

---

## 📊 Example Output
![Prediction Plot](../assets/prediction_plot.png)

Example output shows how the model compares true and predicted trajectories for a selected feature over the next three time steps.

## Key Idea
This project explores **how interventions influence the future trajectory of a dynamic system**. Instead of predicting only the next state, the model predicts multiple future steps and compares outcomes under different intervention scenarios.

This enables rapid exploration of questions like:
- What would happen under no intervention?
- How does a treatment alter the trajectory over time?

## Motivation
Understanding how biological systems evolve over time is a fundamental problem in computational biology. Many real-world scenarios require predicting system behavior under different intervention conditions — whether for drug efficacy assessment, treatment response prediction, or understanding disease progression.

Traditional approaches often treat prediction and intervention effects separately. BioSim integrates both in a unified framework: given the current state and a proposed intervention, the model predicts how the system will evolve.

This capability is valuable because:
- **Intervention effects are state-dependent:** A drug's effect depends on the current biological context
- **Multi-step horizons matter:** Research and clinical decisions often require predictions beyond one time step
- **Computational efficiency:** Trained models allow rapid exploration of scenarios without expensive wet-lab experiments

## Technical Approach
BioSim learns a parameterized dynamical system:

$$\mathbf{x}_{t+k} = f_\theta(\mathbf{x}_t, \mathbf{u}_t; k)$$

where $\mathbf{x}_t$ is the state vector at time $t$, $\mathbf{u}_t$ is an intervention vector, and $k \in \{1, 2, 3\}$ indexes the prediction horizon.

The function $f_\theta$ is implemented as a feedforward neural network trained on synthetic biological trajectories. The model is **intervention-aware**: it conditions predictions explicitly on intervention input, allowing it to capture how system dynamics change under external perturbations.

## What This Project Demonstrates
BioSim is a working prototype that addresses the core challenge of predicting time-evolving biological systems while modeling intervention effects. Using synthetic trajectories, it shows that:

1. Intervention effects can be learned from trajectory data
2. Multi-step predictions are feasible with a simple neural architecture
3. Intervention-aware models can compare different intervention scenarios

## Data and Model
**Input:**
- Current state vector: 5 continuous features (biological measurements)
- Intervention vector: 5 continuous features (treatment/external factors)
- Combined input: 10 features total

**Output:**
- Future states at $t+1, t+2, t+3$: 15 features (3 steps × 5 features)
- Predictions are flattened for architectural simplicity

**Architecture:**
- Feedforward neural network with two hidden layers (32 neurons each, ReLU activation)
- Input size: 10 | Hidden: 32 → 32 | Output: 15
- MSE loss for training
- Adam optimizer with learning rate 0.001

## Project Files
```
BioSim/
├── requirements.txt          # Python dependencies
├── src/
│   ├── load_data.py          # Synthetic trajectory generation with interventions
│   ├── model.py              # Neural network architecture (BioSimModel)
│   ├── train.py              # Training pipeline
│   └── predict.py            # Inference and visualization
└── README.md
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
cd src
python train.py
```

This trains on 1000 synthetic trajectories for 50 epochs and saves the model to `biosim_model.pt`.

### Inference and Visualization
```bash
python predict.py
```

This runs inference on a sample and generates:
- Prediction accuracy metrics (MAE per time step)
- True vs predicted trajectory visualization
- Intervention effect analysis

## Results Interpretation
The generated plots show:
- **X-axis:** Prediction horizon (t+1, t+2, t+3)
- **Y-axis:** Feature value for the selected dimension
- **Blue line:** Ground-truth trajectory
- **Red dashed line:** Model predictions

Good predictions should follow the true trajectory closely across the horizon.

## Limitations and Current Scope
This is a research prototype with intentional simplifications:

- **Synthetic data:** Uses generated trajectories, not empirical measurements
- **Simplified dynamics:** Ground truth follows a simple autoregressive process with interventions
- **Small-scale:** 5-dimensional state space; real biology is higher-dimensional
- **No uncertainty quantification:** Deterministic predictions only
- **Limited validation:** No train/test split or cross-validation
- **Single architecture:** Fixed network size; no architecture search

These limitations are addressed in the Future Work section.

## Future Work
**Near-term (Model Improvement):**
1. **Sequence modeling:** Replace feedforward architecture with LSTM/GRU to better capture temporal dependencies
2. **Uncertainty quantification:** Use probabilistic outputs (e.g., mixture density networks or Bayesian approaches)
3. **Longer horizons:** Extend prediction beyond 3 steps and compare recursive vs direct strategies

**Medium-term (Experimental Validation):**
4. **Real data integration:** Benchmark with published biological time-series datasets (e.g., TCGA, GTEx)
5. **Intervention discovery:** Learn to recommend optimal interventions using reinforcement learning or optimal control

**Long-term (Systems Understanding):**
6. **Mechanistic interpretability:** Decompose learned dynamics into putative biological subsystems
7. **Generalization across conditions:** Train on diverse intervention types and test out-of-distribution robustness
8. **Multi-scale modeling:** Integrate predictions across molecular, cellular, and organismal scales

---

**Citation and Attribution:**
If you use BioSim in research, please reference this repository. This work was developed as a proof-of-concept for intervention-aware dynamical systems modeling in biology.

For questions or collaborations, please open an issue or contact the authors.

*Last updated: April 2026*

<parameter name="filePath">c:\Users\prana\BioSim\README.md

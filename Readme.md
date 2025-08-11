# Quantum Variational Classifier

A quantum machine learning implementation that uses variational quantum circuits to classify Iris dataset samples. This project compares two optimization approaches: COBYLA (classical) and SPSA (Simultaneous Perturbation Stochastic Approximation) for training quantum classifiers.

## Overview

This project implements a quantum binary classifier using Qiskit that:
- Encodes classical data into quantum states using rotation gates
- Uses a parameterized quantum circuit (PQC) as a variational quantum classifier
- Optimizes circuit parameters using classical optimization algorithms
- Performs hyperparameter tuning with Optuna
- Evaluates performance on the Iris dataset (binary classification: setosa vs versicolor)

## Features

- **Quantum Data Encoding**: Maps classical features to quantum states using RY rotation gates
- **Variational Quantum Circuit**: 2-qubit parameterized circuit with entangling gates
- **Multiple Optimizers**: Comparison between COBYLA and SPSA optimization methods
- **Hyperparameter Tuning**: Automated optimization using Optuna framework
- **Performance Visualization**: Scatter plots showing classification results
- **Multi-criteria Evaluation**: Ranking based on accuracy, runtime, penalty, and parameter magnitude

## Requirements

```python
numpy
matplotlib
scikit-learn
scipy
optuna
qiskit
qiskit-aer
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Simply run the script:

```bash
python smart Classifier.py
```

The script will:
1. Load and preprocess the Iris dataset
2. Apply PCA dimensionality reduction to 2 features
3. Split data into training and test sets
4. Tune hyperparameters for both COBYLA and SPSA optimizers
5. Train quantum classifiers with optimal parameters
6. Evaluate and visualize results

## Architecture

### Quantum Circuit Structure

The variational quantum circuit consists of:
- **Data Encoding Layer**: RY rotations encoding input features
- **Variational Layer**: Parameterized RY rotations with learnable parameters
- **Entangling Layer**: CNOT gate creating quantum correlations
- **Final Variational Layer**: Additional parameterized rotations

### Classification Method

Classification is performed by:
1. Measuring the quantum circuit in the computational basis
2. Computing probability difference: P(|00⟩ + |01⟩) - P(|10⟩ + |11⟩)
3. Assigning class based on the sign of the probability difference

### Optimization Approaches

**COBYLA (Constrained Optimization BY Linear Approximation)**
- Classical derivative-free optimization
- Good for noisy objective functions
- Deterministic approach

**SPSA (Simultaneous Perturbation Stochastic Approximation)**
- Gradient-free stochastic optimization
- Estimates gradients using function evaluations
- Particularly suited for quantum optimization problems

## Hyperparameter Tuning

The project uses Optuna for automated hyperparameter optimization:

### COBYLA Parameters
- `maxiter`: Maximum number of iterations (50-200)
- `init_param_*`: Initial circuit parameters (-π to π)

### SPSA Parameters
- `a0`: Initial step size (0.05-0.3)
- `c0`: Initial perturbation magnitude (0.05-0.3)
- `nsteps`: Number of optimization steps (50-200)
- `init_param_*`: Initial circuit parameters (-π to π)

### Multi-criteria Ranking

Trials are ranked by:
1. **Accuracy** (primary, maximized)
2. **Runtime** (minimized)
3. **Penalty** (minimized, for moderate hyperparameter values)
4. **Parameter Magnitude** (minimized, for regularization)

## Results

The script outputs:
- Best hyperparameters for each optimization method
- Optimized quantum circuit parameters
- Test accuracy for both methods
- Runtime and penalty metrics
- Visualization plots comparing true vs predicted classifications

## Key Functions

- `build_circuit(features, params)`: Constructs the parameterized quantum circuit
- `model_predict_scalar(params)`: Makes predictions using statevector simulation
- `classify(params, X_data)`: Performs classification using shot-based simulation
- `train_cobyla(init_params, maxiter)`: Trains using COBYLA optimizer
- `train_spsa(init_params, nsteps, a0, c0)`: Trains using SPSA optimizer
- `rank_trials(study)`: Multi-criteria ranking of optimization trials

## Quantum Advantage

This implementation demonstrates:
- **Quantum Feature Maps**: Encoding classical data in quantum Hilbert space
- **Quantum Entanglement**: Using quantum correlations for classification
- **Variational Learning**: Optimizing quantum circuits for machine learning tasks

## Extensions

Potential improvements and extensions:
- Multi-class classification using one-vs-all or multi-output circuits
- More complex quantum circuits with additional layers
- Hardware-aware optimization for NISQ devices
- Comparison with classical machine learning baselines
- Integration with quantum feature maps and kernels

## References

- [Qiskit Machine Learning](https://qiskit.org/ecosystem/machine-learning/)
- [Variational Quantum Classifiers](https://quantum-computing.ibm.com/lab/docs/iql/machine-learning)
- [SPSA Optimization](https://www.jhuapl.edu/SPSA/)


## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
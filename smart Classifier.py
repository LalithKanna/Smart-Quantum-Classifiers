import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
import optuna
from qiskit import QuantumCircuit
from qiskit_aer import StatevectorSimulator, AerSimulator
import time

# -------------------------
# Data preparation
# -------------------------
iris = datasets.load_iris()
X, y = iris.data, iris.target
mask = (y == 0) | (y == 1)  # binary classification
X, y = X[mask], y[mask]
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)
y_train_ev = 2 * y_train - 1
y_test_ev = 2 * y_test - 1

# Simulators
sim_train = StatevectorSimulator(seed_simulator=42)
sim_predict = AerSimulator(seed_simulator=42, shots=1024)

# -------------------------
# Quantum circuit builder
# -------------------------
def build_circuit(features, params):
    qc = QuantumCircuit(2)
    qc.ry(float(features[0]), 0)
    qc.ry(float(features[1]), 1)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 0)
    qc.ry(params[3], 1)
    return qc

# -------------------------
# Prediction helpers
# -------------------------
def all_basis_probabilities(sv):
    return np.abs(sv.data) ** 2

def probability_difference(sv):
    probs = all_basis_probabilities(sv)
    return (probs[0] + probs[1]) - (probs[2] + probs[3])

def model_predict_scalar(params):
    preds = []
    for x in X_train:
        qc = build_circuit(x, params)
        sv = sim_train.run(qc).result().get_statevector()
        preds.append(probability_difference(sv))
    return np.array(preds)

def classify(params, X_data):
    preds = []
    for x in X_data:
        qc = build_circuit(x, params)
        qc.measure_all()  # Required for AerSimulator

        result = sim_predict.run(qc).result()
        counts = result.get_counts()

        counts_00 = counts.get('00', 0)
        counts_01 = counts.get('01', 0)
        counts_10 = counts.get('10', 0)
        counts_11 = counts.get('11', 0)
        total_counts = sum(counts.values())

        # Calculate probability difference from counts
        prob_diff = (counts_00 + counts_01 - counts_10 - counts_11) / total_counts
        preds.append(np.sign(prob_diff))

    return ((np.array(preds) + 1) // 2).astype(int)

def loss_fn_from_preds(preds, labels_ev):
    return np.mean((preds - labels_ev) ** 2)

# -------------------------
# SPSA optimizer step
# -------------------------
def spsa_step(params, a, c):
    delta = 2 * (np.random.rand(len(params)) > 0.5).astype(float) - 1.0
    p_plus = params + c * delta
    p_minus = params - c * delta
    preds_p = model_predict_scalar(p_plus)
    preds_m = model_predict_scalar(p_minus)
    loss_p = loss_fn_from_preds(preds_p, y_train_ev)
    loss_m = loss_fn_from_preds(preds_m, y_train_ev)
    g_hat = (loss_p - loss_m) / (2 * c * delta)
    params_new = params - a * g_hat
    return params_new

# -------------------------
# Training functions
# -------------------------
def train_cobyla(init_params, maxiter=80):
    def obj(p):
        preds = model_predict_scalar(p)
        loss = loss_fn_from_preds(preds, y_train_ev)
        return float(loss)
    res = minimize(obj, init_params, method='COBYLA', options={'maxiter': maxiter, 'disp': False})
    return res.x

def train_spsa(init_params, nsteps=80, a0=0.1, c0=0.1):
    params = init_params.copy()
    losses = []
    for k in range(1, nsteps+1):
        a = a0/(k**0.602)
        c = c0/(k**0.101)
        params = spsa_step(params, a, c)
        preds = model_predict_scalar(params)
        loss = loss_fn_from_preds(preds, y_train_ev)
        losses.append(loss)
        if k % 10 == 0:
            print(f"SPSA step {k}/{nsteps}, Loss: {loss:.4f}")
    return params, losses

# -------------------------
# Optuna objective functions
# -------------------------
def objective_cobyla(trial):
    maxiter = trial.suggest_int("maxiter", 50, 200)
    init_params = np.array([
        trial.suggest_uniform(f"init_param_{i}", -np.pi, np.pi) for i in range(4)
    ])

    start_time = time.time()
    opt_params = train_cobyla(init_params, maxiter=maxiter)
    runtime = time.time() - start_time

    y_pred = classify(opt_params, X_test)
    accuracy = accuracy_score(y_test, y_pred)

    param_magnitude = np.linalg.norm(opt_params)
    trial.set_user_attr("runtime", runtime)
    trial.set_user_attr("param_magnitude", param_magnitude)
    trial.set_user_attr("penalty", 0)
    trial.set_user_attr("a0", None)
    trial.set_user_attr("c0", None)
    trial.set_user_attr("opt_params", opt_params.tolist())  # Save params

    return accuracy

def objective_spsa(trial):
    a0 = trial.suggest_float("a0", 0.05, 0.3)
    c0 = trial.suggest_float("c0", 0.05, 0.3)
    nsteps = trial.suggest_int("nsteps", 50, 200)
    init_params = np.array([
        trial.suggest_uniform(f"init_param_{i}", -np.pi, np.pi) for i in range(4)
    ])

    start_time = time.time()
    opt_params, losses = train_spsa(init_params, nsteps=nsteps, a0=a0, c0=c0)
    runtime = time.time() - start_time

    y_pred = classify(opt_params, X_test)
    accuracy = accuracy_score(y_test, y_pred)

    param_magnitude = np.linalg.norm(opt_params)

    def moderate_penalty(x):
        if 0.1 <= x <= 0.2:
            return 0
        else:
            return min(abs(x - 0.1), abs(x - 0.2))

    penalty_a0 = moderate_penalty(a0)
    penalty_c0 = moderate_penalty(c0)
    penalty = penalty_a0 + penalty_c0

    trial.set_user_attr("runtime", runtime)
    trial.set_user_attr("penalty", penalty)
    trial.set_user_attr("param_magnitude", param_magnitude)
    trial.set_user_attr("a0", a0)
    trial.set_user_attr("c0", c0)
    trial.set_user_attr("opt_params", opt_params.tolist())  # Save params
    trial.set_user_attr("losses", losses)

    return accuracy

# -------------------------
# Rank trials by multiple criteria
# -------------------------
def rank_trials(study):
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(
        completed_trials,
        key=lambda t: (
            -t.value,  # max accuracy
            t.user_attrs.get("runtime", float('inf')),
            t.user_attrs.get("penalty", float('inf')),
            t.user_attrs.get("param_magnitude", float('inf'))
        )
    )
    return sorted_trials[0]

# -------------------------
# Run Optuna tuning
# -------------------------
print("Tuning COBYLA...")
study_cobyla = optuna.create_study(direction="maximize")
study_cobyla.optimize(objective_cobyla, n_trials=20)
best_cobyla_trial = rank_trials(study_cobyla)
print("Best COBYLA params:", best_cobyla_trial.params)
print("Optimized circuit parameters:", best_cobyla_trial.user_attrs["opt_params"])
print("Accuracy:", best_cobyla_trial.value)
print("Runtime:", best_cobyla_trial.user_attrs["runtime"])
print("Penalty (moderate param):", best_cobyla_trial.user_attrs["penalty"])
print("Param magnitude:", best_cobyla_trial.user_attrs["param_magnitude"])
print("a0, c0:", best_cobyla_trial.user_attrs["a0"], best_cobyla_trial.user_attrs["c0"])

print("\nTuning SPSA...")
study_spsa = optuna.create_study(direction="maximize")
study_spsa.optimize(objective_spsa, n_trials=20)
best_spsa_trial = rank_trials(study_spsa)
print("Best SPSA params:", best_spsa_trial)
# Predict with best COBYLA params and evaluate
best_cobyla_params = np.array(best_cobyla_trial.user_attrs["opt_params"])
y_pred_cobyla = classify(best_cobyla_params, X_test)
acc_cobyla = accuracy_score(y_test, y_pred_cobyla)
print(f"COBYLA Test Accuracy: {acc_cobyla:.4f}")

# Predict with best SPSA params and evaluate
best_spsa_params = np.array(best_spsa_trial.user_attrs["opt_params"])
y_pred_spsa = classify(best_spsa_params, X_test)
acc_spsa = accuracy_score(y_test, y_pred_spsa)
print(f"SPSA Test Accuracy: {acc_spsa:.4f}")

# Visualization function
def plot_results(X, y_true, y_pred, title):
    plt.figure(figsize=(6,5))
    for lbl in [0, 1]:
        plt.scatter(X[y_true == lbl, 0], X[y_true == lbl, 1], label=f"True {lbl}", alpha=0.6)
    for lbl in [0, 1]:
        idx = np.where(y_pred == lbl)[0]
        plt.scatter(X[idx, 0], X[idx, 1], marker='x', label=f"Predicted {lbl}")
    plt.title(title)
    plt.legend()
    plt.show()

# Plot results for COBYLA
plot_results(X_test, y_test, y_pred_cobyla, "COBYLA Classifier Predictions")

# Plot results for SPSA
plot_results(X_test, y_test, y_pred_spsa, "SPSA Classifier Predictions")


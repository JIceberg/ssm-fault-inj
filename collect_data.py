import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

plt.rcParams.update({
    'font.size': 16,        # Base font size for labels and ticks
    'font.weight': 'bold',  # Make fonts bold
    'axes.labelweight': 'bold',  # Bold axis labels
    'axes.titlesize': 20,       # Title font size
    'axes.titleweight': 'bold',
    'legend.fontsize': 14,      # Legend font size
    'xtick.labelsize': 14,      # Tick labels
    'ytick.labelsize': 14,
})

def extract_mean_acc(data, prefix):
    error_rates = []
    mean_accs = []

    prefix_str = prefix + "-"

    for key, value in data.items():
        if key.startswith(prefix_str):
            error_rate = float(key[len(prefix_str):])
            mean_acc = np.mean(value["acc"]) * 100  # convert to percentage
            error_rates.append(error_rate)
            mean_accs.append(mean_acc)

    pairs = sorted(zip(error_rates, mean_accs))
    if pairs:
        error_rates, mean_accs = zip(*pairs)
    else:
        error_rates, mean_accs = [], []

    return error_rates, mean_accs


# -------- ARGUMENTS --------
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, nargs='+',
                    help="Model name(s), e.g., mnist cifar10. Use 'all' for all models.")
parser.add_argument("--injection", required=True, help="Injection type")
args = parser.parse_args()

all_models = ["mnist", "cifar10", "cifar100"]
models = all_models if "all" in args.model else args.model

nominal_acc = {
    "mnist": 95.27,
    "cifar10": 83.45,
    "cifar100": 63.05
}

def compute_recovery_power(method_acc, nocorr_acc, nominal_acc_model):
    """
    method_acc, nocorr_acc: list of accuracies (%) for different error rates
    nominal_acc_model: nominal accuracy (%) for the model
    returns: list of recovery power values
    """
    method_acc = np.array(method_acc)
    nocorr_acc = np.array(nocorr_acc)
    rp = (method_acc - nocorr_acc) / (nominal_acc_model - nocorr_acc)
    return rp

def weighted_average_recovery_power(rp, error_rates):
    weights = error_rates / np.sum(error_rates)
    return np.sum(rp * weights)

# -------- CREATE SUBPLOTS --------
n_models = len(models)

fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5), sharey=False)

if n_models == 1:  # axes is not a list if only one subplot
    axes = [axes]

fig.subplots_adjust(wspace=1.5)  # adjust as needed

for idx, (ax, model) in enumerate(zip(axes, models)):
    zeroing_file = f"{model}_stats_{args.injection}_zeroing.json"
    bit2_file = f"{model}_stats_{args.injection}_2bit.json"
    bit4_file = f"{model}_stats_{args.injection}_4bit.json"

    if not (os.path.exists(zeroing_file) and os.path.exists(bit2_file) and os.path.exists(bit4_file)):
        print(f"Skipping {model}: one of the files is missing.")
        continue

    with open(zeroing_file) as f:
        zeroing_data = json.load(f)
    with open(bit4_file) as f:
        bit4_data = json.load(f)
    with open(bit2_file) as f:
        bit2_data = json.load(f)

    zero_x, zero_y = extract_mean_acc(zeroing_data, "zeroing")
    nocorr_x, nocorr_y = extract_mean_acc(bit4_data, "no-corr")
    backup4_x, backup4_y = extract_mean_acc(bit4_data, "backup")
    backup2_x, backup2_y = extract_mean_acc(bit2_data, "backup")

    # Compute recovery power for each method
    nom_acc = nominal_acc[model]
    zero_rp = compute_recovery_power(zero_y, nocorr_y, nom_acc)
    backup4_rp = compute_recovery_power(backup4_y, nocorr_y, nom_acc)
    backup2_rp = compute_recovery_power(backup2_y, nocorr_y, nom_acc)

    # Compute average recovery power across all error rates
    avg_zero_rp = weighted_average_recovery_power(zero_rp, zero_x)
    avg_backup4_rp = weighted_average_recovery_power(backup4_rp, backup4_x)
    avg_backup2_rp = weighted_average_recovery_power(backup2_rp, backup2_x)

    # Print results
    print(f"Model: {model.upper()}")
    print(f"  Zeroing: {avg_zero_rp:.3f}")
    print(f"  Backup (4-bit): {avg_backup4_rp:.3f}")
    print(f"  Backup (2-bit): {avg_backup2_rp:.3f}")

    # Plot curves
    ax.plot(nocorr_x, nocorr_y, color='red', marker='o', markersize=10,
            markerfacecolor='red', linewidth=3, label="No Correction")
    ax.plot(zero_x, zero_y, color='orange', marker='^', markersize=10,
            markerfacecolor='orange', linewidth=3, label="Output Suppression")
    ax.plot(backup4_x, backup4_y, color='green', marker='s', markersize=10,
            markerfacecolor='green', linewidth=3, label="Child (4bit)")
    ax.plot(backup2_x, backup2_y, color='purple', marker='D', markersize=6,
            markerfacecolor='purple', linewidth=3, label="Child (2bit)")

    ax.set_xscale("log")
    ax.set_xlabel("Error Rate")
    if idx == 0:
        ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{model}".upper())
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
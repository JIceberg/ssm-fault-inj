import json
import re
import statistics
import matplotlib.pyplot as plt
from collections import defaultdict

import matplotlib.pyplot as plt

# Example data
sparsity = [30, 40, 50, 60, 70, 80]
accuracy = [0.8553, 0.8615, 0.8885, 0.8973, 0.9072, 0.9185]

plt.figure()
plt.plot(sparsity, accuracy, marker='o')
plt.xlabel("Sparsity (%)")
plt.ylabel("Accuracy")
plt.grid(True)

plt.show()

with open("mnist_stats_out.json", "r") as f:
    raw_data = json.load(f)

results = {}

for k, v in raw_data.items():
    acc_mean = statistics.mean(v["acc"])
    time_median = statistics.median(v["time"])
    if "gradient" in k:
        continue
    results[k] = {
        "acc": acc_mean,
        "time": time_median
    }

for key, vals in results.items():
    print(f"{key}: mean(acc)={vals['acc']:.3f}, median(time)={vals['time']:.3f}")

error_rates = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

all_acc = {k: v["acc"] for k, v in results.items()}
nominal_acc = all_acc["nominal"]

methods = {}

sci_pat = re.compile(r'\b[0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+\b')
dec_pat = re.compile(r'\b[0-9]*\.[0-9]+\b')

def normalize_method(method: str):
    if method.startswith("backup-clean-corr") or method.startswith("backup-dirty-corr"):
        return "backup-delta"
    if method.startswith("backup-clean-checksum-corr") or method.startswith("backup-dirty-checksum-corr"):
        return "backup-checksum"
    return method

def parse_key(key: str):
    # Try scientific notation first
    m = sci_pat.search(key)

    # If not found, fallback to decimal
    if not m:
        m = dec_pat.search(key)

    if not m:
        raise ValueError(f"Could not find an error-rate in key: {key}")

    s, e = m.span()

    # Find hyphens on both sides
    hyph_left = key.rfind("-", 0, s)
    hyph_right = key.find("-", e)

    if hyph_left == -1 or hyph_right == -1:
        raise ValueError(f"Unexpected format in key: {key}")

    method = key[:hyph_left]
    err_rate = float(key[s:e])
    suffix = key[hyph_right+1:]

    return method, err_rate, suffix

median_times = defaultdict(dict)

median_times = defaultdict(lambda: defaultdict(list))

for key, vals in results.items():
    if key == "nominal":
        continue

    try:
        method, _, suffix = parse_key(key)
    except ValueError:
        continue

    method = normalize_method(method)
    median_time = vals["time"]
    median_times[method][suffix].append(median_time)
    
avg_median_times = {}

for method, suffix_dict in median_times.items():
    # Average over error rates per suffix
    suffix_avg = [sum(times)/len(times) for times in suffix_dict.values()]
    # Average across suffixes
    avg_median_times[method] = sum(suffix_avg) / len(suffix_avg)

method_names = {
    "no-corr": "no correction",
    "external-zero-corr": "external grad",
    "internal-zero-corr": "internal grad",
    "backup-delta": "backup,\ndelta detection",
    "backup-checksum": "backup,\nchecksum detection",
    "nominal": "nominal"
}

methods = list(avg_median_times.keys())
times = [avg_median_times[m] for m in methods]
methods = [method_names[m] for m in methods]

plt.figure(figsize=(10,6))
bars = plt.bar(methods, times, color='skyblue')

plt.ylabel("Inference Latency (ms)")
plt.xlabel("Correction Method")
plt.xticks(rotation=0, ha='center')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Optional: add numeric labels on top of bars
for bar, time in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{time:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

data = defaultdict(lambda: defaultdict(dict))

for key, acc in all_acc.items():
    if key == "nominal":
        continue
    method, err, suffix = parse_key(key)
    data[method][suffix][err] = acc

def plot_injection_type(data, suffix, error_rates, nominal_acc):
    fig, ax = plt.subplots(figsize=(8, 6))

    method_names = {
    "no-corr": "no correction",
    "external-zero-corr": "external gradient zeroing",
    "internal-zero-corr": "internal gradient zeroing",
    "backup-clean-corr": "error-free backup, delta detection",
    "backup-dirty-corr": "error-prone backup, delta detection",
    "backup-clean-checksum-corr": "error-free backup, checksum detection",
    "backup-dirty-checksum-corr": "error-prone backup, checksum detection",
    "nominal": "nominal"
    }

    for method, suffix_dict in data.items():
        if suffix not in suffix_dict:
            continue

        accuracies = [suffix_dict[suffix].get(e, None) for e in error_rates]

        # # Visual hierarchy (optional but recommended)
        # if method in {"no-corr", "external-zero-corr"}:
        #     alpha = 0.4
        #     lw = 1.8
        #     z = 1
        # else:
        #     alpha = 0.9
        #     lw = 3.0
        #     z = 3

        z = 3 if "backup" in method else 1

        # Line style by family
        if "checksum" in method:
            ls = ":"
            marker = "o"
        elif "backup-clean" in method:
            ls = "--"
            marker = "o"
        elif "backup-dirty" in method:
            ls = "-."
            marker = "o"
        else:
            ls = "-"
            marker = "D"

        ax.plot(
            error_rates,
            accuracies,
            linestyle=ls,
            marker=marker,
            # linewidth=lw,
            # alpha=alpha,
            label=method_names[method],
            zorder=z
        )

    # Nominal baseline
    ax.plot(
        error_rates,
        [nominal_acc] * len(error_rates),
        linestyle="--",
        color="black",
        linewidth=2,
        label="nominal",
        zorder=0
    )

    ax.set_xscale("log")
    ax.set_xlabel("Error rate")
    ax.set_ylabel("Accuracy")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


# --- Create separate figures ---
plot_injection_type(data, "output", error_rates, nominal_acc)
plot_injection_type(data, "weight", error_rates, nominal_acc)
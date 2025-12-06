import json
import re
import statistics
import matplotlib.pyplot as plt
from collections import defaultdict

with open("stats_out.json", "r") as f:
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

    median_time = vals["time"]
    median_times[method][suffix].append(median_time)
    
avg_median_times = {}

for method, suffix_dict in median_times.items():
    # Average over error rates per suffix
    suffix_avg = [sum(times)/len(times) for times in suffix_dict.values()]
    # Average across suffixes
    avg_median_times[method] = sum(suffix_avg) / len(suffix_avg)

methods = list(avg_median_times.keys())
times = [avg_median_times[m] for m in methods]

plt.figure(figsize=(10,6))
bars = plt.bar(methods, times, color='skyblue')

plt.ylabel("Median Execution Time (ms)")
plt.xlabel("Injection Method")
plt.title("Comparison of Median Times Across Methods")
plt.xticks(rotation=45, ha='right')
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

for method, suffix_dict in data.items():
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(f"Performance for correction method: {method}")

    for ax, suffix in zip(axes, ["output", "weight"]):
        if suffix not in suffix_dict:
            ax.set_title(f"No data for injection type '{suffix}'")
            continue

        # accuracy values for this suffix
        accuracies = [suffix_dict[suffix].get(e, None) for e in error_rates]

        # Plot nominal (horizontal)
        ax.plot(error_rates, [nominal_acc]*len(error_rates),
                label="nominal", linestyle="-", color="black")

        # Plot method
        ax.plot(error_rates, accuracies, marker="o", label=f"{method}-{suffix}")

        ax.set_xscale("log")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.set_title(f"Injection type: {suffix}")

    axes[-1].set_xlabel("Error rate (log scale)")
    plt.tight_layout()
    plt.show()
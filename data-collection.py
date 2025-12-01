import matplotlib.pyplot as plt
import numpy as np

# Top chart data
nominal = [0.7967, 0.7967, 0.7967]
faulty_no_corr = [0.6417, 0.6446, 0.6398]
faulty_grad_zero_external = [0.7269, 0.7281, 0.7311]
faulty_grad_zero_internal = [0.7965, 0.7966, 0.7966]
faulty_backup_clean = [0.7306, 0.7295, 0.7304]
faulty_backup_faulty = [0.7476, 0.7471, 0.7493]
faulty_backup_clean_checksum = [0.7889, 0.7883, 0.7901]
faulty_backup_faulty_chcksum = [0.7921, 0.7912, 0.7915]

averages_top = [
    np.mean(nominal),
    np.mean(faulty_no_corr),
    np.mean(faulty_grad_zero_external),
    np.mean(faulty_grad_zero_internal),
    np.mean(faulty_backup_clean),
    np.mean(faulty_backup_faulty),
    np.mean(faulty_backup_clean_checksum),
    np.mean(faulty_backup_faulty_chcksum)
]

labels_top = [
    "Nominal",
    "No Correction",
    "Zeroing (External)",
    "Zeroing (Internal)",
    "Backup (Clean)",
    "Backup (Faulty)",
    "Backup (Clean) Checksum",
    "Backup (Faulty) Checksum"
]

# Bottom chart data
faulty_no_corr_vals = [0.2764, 0.2988, 0.3037]
faulty_ext_vals = [0.5522, 0.5139, 0.5368]
faulty_int_vals = [0.7963, 0.7971, 0.7967]
backup_clean_vals = [0.7291, 0.7308, 0.7299]
backup_faulty_vals = [0.4112, 0.4836, 0.4703]
backup_clean_checksum_vals = [0.3884, 0.3045, 0.3468]
backup_faulty_checksum_vals = [0.3748, 0.4461, 0.3417]

averages_bottom = [
    np.mean(nominal),
    np.mean(faulty_no_corr_vals),
    np.mean(faulty_ext_vals),
    np.mean(faulty_int_vals),
    np.mean(backup_clean_vals),
    np.mean(backup_faulty_vals),
    np.mean(backup_clean_checksum_vals),
    np.mean(backup_faulty_checksum_vals)
]

labels_bottom = [
    "Nominal",
    "No Correction",
    "Zeroing (External)",
    "Zeroing (Internal)",
    "Backup (Clean)",
    "Backup (Faulty)",
    "Backup (Clean) Checksum",
    "Backup (Faulty) Checksum"
]

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

# Top chart
ax1.bar(labels_top, averages_top)
ax1.set_ylabel("Average Accuracy")
ax1.set_title("Average Accuracy with Injection into Output")
ax1.set_ylim(0.5, 0.85)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.tick_params(axis='x', rotation=90)

# Bottom chart
ax2.bar(labels_bottom, averages_bottom)
ax2.set_ylabel("Average Accuracy")
ax2.set_title("Average Accuracy with Injection into Weight")
ax2.set_ylim(0.2, 0.85)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

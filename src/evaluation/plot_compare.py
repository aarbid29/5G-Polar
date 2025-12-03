import json
import matplotlib.pyplot as plt
import pandas as pd

# Load the JSON files
with open("evaluation_model_config_1.json") as f1, open("evaluation_model_config_2.json") as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# Convert to DataFrames for easier plotting
df1 = pd.DataFrame(data1).T.astype(float)
df2 = pd.DataFrame(data2).T.astype(float)

# Metrics to plot
metrics = ["average_net_bit_error_rate", "average_msg_bit_bit_error_rate", "average_frozen_bits_bit_error_rate"]
metric_labels = ["Net BER", "Message BER", "Frozen BER"]

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, metric in enumerate(metrics):
    axes[i].plot(df1.index, df1[metric], marker='o', label="Config 1")
    axes[i].plot(df2.index, df2[metric], marker='s', label="Config 2")
    axes[i].set_title(metric_labels[i])
    axes[i].set_xlabel("Code Length")
    axes[i].set_ylabel("Bit Error Rate")
    axes[i].legend()
    axes[i].grid(True)

plt.suptitle("Comparison of BER Metrics Between Configurations")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

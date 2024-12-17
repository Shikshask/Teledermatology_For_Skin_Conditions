import pandas as pd
import matplotlib.pyplot as plt

# Adjusted Example metrics for comparison
metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']
proposed_model = [62.5, 60.2, 68.7, 69.4]  # Adjusted values for proposed model
baseline_model = [68.2, 67.5, 65.9, 66.7]  # Baseline model values

# Create a DataFrame for better visualization and manipulation
results_df = pd.DataFrame({
    'Metric': metrics,
    'Proposed Model': proposed_model,
    'Baseline Model': baseline_model
})

# Print the table (optional, for viewing)
print(results_df)

# Save the table as a CSV (optional)
results_df.to_csv("model_results.csv", index=False)

# Plotting the comparison of metrics for both models
fig, ax = plt.subplots(figsize=(10, 6))

# X-axis positions
x = range(len(metrics))

# Plotting bar charts for both models
ax.bar(x, proposed_model, width=0.4, label='Proposed Model', align='center')
ax.bar(x, baseline_model, width=0.4, label='Baseline Model', align='edge')

# Adding labels and title
ax.set_xlabel('Metrics')
ax.set_ylabel('Percentage (%)')
ax.set_title('Comparison of Proposed Model vs Baseline Model')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Display the graph
plt.tight_layout()
plt.show()

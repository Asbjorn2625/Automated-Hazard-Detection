import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Get current folder
cur_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(cur_dir, "Testing the U-net/results.csv")

# Load the CSV file
df = pd.read_csv(csv_path)

# Drop the 'image' column
df = df.drop(columns=['image'])

# Now group by 'mask' and 'model', and take the mean of the other columns
grouped_df = df.groupby(['mask', 'model']).mean()

# Reset the index to make 'mask' and 'model' back to columns
grouped_df = grouped_df.reset_index()

# Now, grouped_df is your result. You can print it out
print(grouped_df)

# Ensure that the data types are appropriate for plotting
grouped_df[['precision', 'recall', 'accuracy', 'dice', 'iou']] = grouped_df[['precision', 'recall', 'accuracy', 'dice', 'iou']].apply(pd.to_numeric)

# Metrics we want to plot
metrics = ['precision', 'recall', 'accuracy', 'dice', 'iou']

models = grouped_df['model'].unique()
masks = grouped_df['mask'].unique()

bar_width = 0.1
x = np.arange(len(masks))

for metric in metrics:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each model
    for i, model in enumerate(models):
        model_data = grouped_df[grouped_df['model'] == model]
        ax.bar(x + i*bar_width, model_data[metric], width=bar_width, label=model)

    # Fix the x-axes
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(masks)

    ax.set_xlabel('Masks')
    ax.set_ylabel(f'Average {metric.capitalize()}')
    ax.set_title(f'Average {metric.capitalize()} for each Mask-Model pair')
    ax.legend()

    plt.show()

# Round the numeric columns to 3 decimal places
rounded_grouped_df = grouped_df.round(3)

# Convert the DataFrame to a LaTeX table with controlled float format
latex_table = rounded_grouped_df.to_latex(index=False, float_format="%.3f")

# Print the LaTeX table
print(latex_table)
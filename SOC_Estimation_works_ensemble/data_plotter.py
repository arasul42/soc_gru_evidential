import os
import matplotlib.pyplot as plt
from calce_data_plot import FullDataProcessor

def plot_separate_by_source_sorter(df, output_dir="./plots/data_plots"):
    """
    Plots data separated by source_sorter and saves the plots to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    filtered_data = df[df["Step Index"] >= 0]
    source_sorters = filtered_data["source_sorter"].unique()
    colors = plt.cm.tab10.colors

    for sorter in source_sorters:
        sorter_data = filtered_data[filtered_data["source_sorter"] == sorter]

        plt.figure(figsize=(4.72, 3))  # 4.72 inches by 2.15 inches
        plt.plot(sorter_data["Total Time"], sorter_data["Current"], label="Current", color=colors[0], linewidth=0.5)
        plt.plot(sorter_data["Total Time"], sorter_data["Voltage"], label="Voltage", color=colors[1], linewidth=0.5)
        plt.plot(sorter_data["Total Time"], sorter_data["gt_soc"], label="SOC", color=colors[3], linewidth=1)

        plt.title(f"CALCE Dataset: {sorter.replace('_', ' ')}")
        plt.xlabel("Total Time (s)")
        plt.ylabel("Values")
        plt.legend()
        plt.tight_layout()
        plt.grid(True, linestyle='--', linewidth=0.5)

        plot_filename = os.path.join(output_dir, f"CALCE Dataset: {sorter}.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()

# Load data
folder_path = "./training_Set"
full_data = FullDataProcessor(folder_path)
df = full_data.load_data()

# Display the first few rows
print(df.head())

# Call the function after it's defined
plot_separate_by_source_sorter(df)

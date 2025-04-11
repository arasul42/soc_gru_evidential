import os

import matplotlib.pyplot as plt

class BatteryDataPlotter:
    """
    A class to plot battery data, specifically Current, Voltage, and SOC 
    for Step Index 7 grouped by Drain_SOC_Source.
    """
    
    def __init__(self, df):
        """
        Initializes the class with the provided DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing the processed battery data.
        """
        self.df = df

    def plot_step_index_7_by_drain_soc_source(self):
        """
        Plots Current, Voltage, and SOC for Step Index 7 on the same subplot for each Drain_SOC_Source.
        """
        step_index_7_data = self.df[self.df["Step Index"] >=7]  # Filter data for Step Index 7
        drain_soc_sources = step_index_7_data["Drain_SOC_Source"].unique()
        colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

        plt.figure(figsize=(18, 6))

        for i, source in enumerate(drain_soc_sources):
            source_data = step_index_7_data[step_index_7_data["Drain_SOC_Source"] == source]

            plt.subplot(1, len(drain_soc_sources), i + 1)
            plt.plot(source_data["Total Time"], source_data["Current"], label="Current", color=colors[0])
            plt.plot(source_data["Total Time"], source_data["Voltage"], label="Voltage", color=colors[1])
            plt.plot(source_data["Total Time"], source_data["gt_soc"], label="SOC", color=colors[3], linewidth=2)

            plt.title(f"Drain_SOC_Source: {source}")
            plt.xlabel("Total Time (s)")
            plt.ylabel("Values")
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.show()



    def plot_separate_by_drain_soc_source(self, output_dir="./plots/data_plots"):
        """
        Plots Current, Voltage, and SOC for Step Index 7 separately for each Drain_SOC_Source
        and saves the plots in the specified directory.
        
        Args:
            output_dir (str): Directory where the plots will be saved.
        """

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        step_index_7_data = self.df[self.df["Step Index"] >=7]  # Filter data for Step Index 7
        drain_soc_sources = step_index_7_data["Drain_SOC_Source"].unique()
        colors = plt.cm.tab10.colors  # Use the same colormap for consistent colors

        for source in drain_soc_sources:
            source_data = step_index_7_data[step_index_7_data["Drain_SOC_Source"] == source]

            plt.figure(figsize=(10, 5))
            plt.plot(source_data["Total Time"], source_data["Current"], label="Current", color=colors[0])
            plt.plot(source_data["Total Time"], source_data["Voltage"], label="Voltage", color=colors[1])
            plt.plot(source_data["Total Time"], source_data["gt_soc"], label="SOC", color=colors[3], linewidth=2)

            plt.title(f"Dataset {source}")
            plt.xlabel("Total Time (s)")
            plt.ylabel("Values")
            plt.legend()
            plt.grid()

            # Save the plot to the output directory
            plot_filename = os.path.join(output_dir, f"Dataset_{source}.png")
            plt.savefig(plot_filename)
            plt.close()  # Close the plot to free memory



    def plot_separate_by_source_sorter(self, output_dir="./plots/source_sorter_plots"):
        """
        Plots Current, Voltage, and SOC for Step Index 7 separately for each Source_Sorter
        and saves the plots in the specified directory.
        
        Args:
            output_dir (str): Directory where the plots will be saved.
        """

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        step_index_7_data = self.df[self.df["Step Index"] >= 7]  # Filter data for Step Index 7
        source_sorters = step_index_7_data["source_sorter"].unique()
        colors = plt.cm.tab10.colors  # Use the same colormap for consistent colors


        for sorter in source_sorters:
            sorter_data = step_index_7_data[step_index_7_data["source_sorter"] == sorter]

            plt.figure(figsize=(4.72, 3.15))  # 4.72 inches by 3.15 inches
            plt.plot(sorter_data["Total Time"], sorter_data["Current"], label="Current", color=colors[0])
            plt.plot(sorter_data["Total Time"], sorter_data["Voltage"], label="Voltage", color=colors[1])
            plt.plot(sorter_data["Total Time"], sorter_data["gt_soc"], label="SOC", color=colors[3], linewidth=2)

            plt.title(f"Dataset: {sorter.replace('_', ' ')}")
            plt.xlabel("Total Time (s)")
            plt.ylabel("Values")
            plt.legend()
            plt.tight_layout()
            plt.grid(True, linestyle='--', linewidth=0.5)

            # Save the plot to the output directory
            plot_filename = os.path.join(output_dir, f"source_sorter_{sorter}.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close()  # Close the plot to free memory
import os
import glob
import pandas as pd
import re
import argparse

class FullDataProcessor:
    def __init__(self, folder_path, nominal_capacity=2.0):
        """
        Initialize the BatteryDataProcessor class.

        Args:
            folder_path (str): Path to the folder containing data files.
            nominal_capacity (float): The nominal capacity of the battery in Ah (default: 2.0 Ah).
        """
        self.folder_path = folder_path
        self.nominal_capacity = nominal_capacity
        self.valid_sources = ["DST", "FUDS", "BJDST", "US06"]  # Allowed file sources

    def extract_file_details(self, file_name):
        """
        Extracts temperature, file source, and SOC% from the file name.

        Args:
            file_name (str): Name of the file.

        Returns:
            tuple: (temperature in °C, file source as text, drain SOC label e.g., DST_50SOC)
        """
        # Extract temperature (e.g., "0C" or "25C")
        temperature_match = re.search(r"(\d+C)", file_name)
        temperature = int(temperature_match.group()[:-1]) if temperature_match else None

        # Extract file source (DST, FUDS, BJDST, US06) from filename
        file_source_match = re.search(r"_(DST|FUDS|BJDST|US06)_", file_name, re.IGNORECASE)
        file_source = file_source_match.group(1).upper() if file_source_match else None

        # Extract SOC percentage (e.g., 50SOC, 80SOC)
        soc_match = re.search(r"_(\d+SOC)", file_name, re.IGNORECASE)
        soc_percentage = soc_match.group(1).upper() if soc_match else None  # Example: 50SOC, 80SOC
        # Extract SOC number (e.g., 50 from 50SOC)
        soc_number_match = re.search(r"(\d+)SOC", file_name, re.IGNORECASE)
        soc_number = int(soc_number_match.group(1)) if soc_number_match else None



        initial_soc = int(soc_number) / 100 if soc_percentage else 0 # Convert to decimal (e.g., 0.5 for 50SOC)

        # Create Drain_SOC_Source (e.g., DST_50SOC)
        drain_soc_source = f"{file_source}_{soc_percentage}" if file_source and soc_percentage else None

                # Create source_Temperature (e.g., DST_25C)
        source_sorter = f"{drain_soc_source}_{temperature}C" if file_source and temperature is not None else None

        return temperature, file_source, drain_soc_source, initial_soc, source_sorter

    def load_data(self):
        """
        Loads all Excel files from the specified folder, processes them, and returns a combined DataFrame.

        Returns:
            pd.DataFrame: Combined DataFrame containing all processed data.
        """
        file_list = sorted(glob.glob(os.path.join(self.folder_path, "*.xls")))  # Sort files to maintain order
        df_list = []
        total_time = 0  # Accumulated total time

        for file in file_list:
            try:
                # Load file
                df_temp = pd.read_excel(file, sheet_name=1)
                df_temp = df_temp[df_temp["Step_Index"] >= 0]  # Filter data for Step Index 7

                # Extract temperature, file source, and drain SOC source
                temperature, file_source, drain_soc_source, initial_soc, source_sorter = self.extract_file_details(os.path.basename(file))

                # Skip files without a valid file source
                if file_source not in self.valid_sources:
                    print(f"Skipping file {file} - Invalid file source")
                    continue

                df_temp["Temperature (C)"] = temperature
                df_temp["File Source"] = file_source
                df_temp["Drain_SOC_Source"] = drain_soc_source
                df_temp["source_sorter"]=source_sorter  # e.g., DST_50SOC, BJDST_80SOC

                # Rename relevant columns
                df_temp.rename(columns={
                    "Test_Time(s)": "Test Time",
                    "Step_Index": "Step Index",
                    "Current(A)": "Current",
                    "Voltage(V)": "Voltage",
                    "Discharge_Capacity(Ah)": "Discharge Capacity",
                    "Charge_Capacity(Ah)": "Charge Capacity",
                    "Temperature (C)": "Temperature"
                }, inplace=True)

                # Convert to numeric values
                df_temp["Test Time"] = pd.to_numeric(df_temp["Test Time"], errors="coerce")
                df_temp["Step Index"] = pd.to_numeric(df_temp["Step Index"], errors="coerce")
                df_temp["Current"] = pd.to_numeric(df_temp["Current"], errors="coerce")
                df_temp["Voltage"] = pd.to_numeric(df_temp["Voltage"], errors="coerce")
                df_temp["Discharge Capacity"] = pd.to_numeric(df_temp["Discharge Capacity"], errors="coerce")
                df_temp["Charge Capacity"] = pd.to_numeric(df_temp["Charge Capacity"], errors="coerce")
                df_temp["Temperature"] = pd.to_numeric(df_temp["Temperature"], errors="coerce")
                # Compute Ground Truth SOC (gt_soc) using the provided formula
                df_temp["gt_soc"] = 0.0  # Initialize the column with default value 0.0
                gt_soc = [0] # Start with an initial SOC of 0.0

                for i in range(1, len(df_temp)):
                    delta_charge = df_temp["Charge Capacity"].iloc[i] - df_temp["Charge Capacity"].iloc[i - 1]
                    delta_discharge = df_temp["Discharge Capacity"].iloc[i] - df_temp["Discharge Capacity"].iloc[i - 1]

                    # Update SOC based on delta capacity
                    soc = gt_soc[-1] + (delta_charge - delta_discharge) / self.nominal_capacity
                    soc = min(max(soc, 0.0), 1.0)  # Clip between 0 and 1
                    gt_soc.append(soc)

                df_temp["gt_soc"] = gt_soc
                # # Compute Ground Truth SOC (gt_soc)
                # df_temp["gt_soc"] = (self.nominal_capacity - df_temp["Discharge Capacity"]) / self.nominal_capacity
                # df_temp["gt_soc"] = df_temp["gt_soc"].clip(0, 1)  # Clip between 0 and 1



                # Compute Total Time (accumulates across files)
                df_temp["Total Time"] = df_temp["Test Time"] + total_time
                total_time = df_temp["Total Time"].iloc[-1]  # Update total time for next file

                df_list.append(df_temp)

            except Exception as e:
                print(f"Error processing file {file}: {e}")

        # Concatenate all dataframes
        df_combined = pd.concat(df_list, ignore_index=True)
        output_file = "combined_data.xlsx"  # Specify the output file name
        df_combined.to_excel(output_file, index=False)  # Save without the index column
        print(f"Data saved to {output_file}")

        return df_combined


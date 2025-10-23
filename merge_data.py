import pandas as pd
import os

# Folder where your CSVs are stored
folder = "data"  # Change this to your actual folder name
all_data = []

# Loop through each year
for year in range(2015, 2025):
    filename = f"Total Load - Day Ahead _ Actual_{year}01010000-{year+1}01010000.csv"
    filepath = os.path.join(folder, filename)
    
    try:
        print(f"Reading {filename}...")
        df = pd.read_csv(filepath)
        all_data.append(df)
    except FileNotFoundError:
        print(f"⚠️ File not found for {year}: {filename}")
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")

# Merge all dataframes
if all_data:
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_csv("entsoe_10_years_combined.csv", index=False)
    print("✅ All data merged and saved.")
else:
    print("🚫 No data files were successfully read.")

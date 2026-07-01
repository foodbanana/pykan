import pandas as pd
import os

# ================= CONFIGURATION =================
root_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = 'raw_data.xlsx'  # Replace with your actual raw file name

# 1. Define Input Features
FEATURE_COLS = [
    'Current density (A/cm2)',
    'Voltage (V)',
    'Faradaic efficiency',
    'CO conversion',
    'Crossover rate',
    'Capture energy (MJ/kgCO2)',
    'Electricity price (USD/kWh)',
    'Membrane cost (USD/m2)'
]

# 2. Define Targets and Output Filenames
TARGET_MAPPING = {
    'CO2 in cathode (mol frac)': 'CO2RRCC',
    'CO2 in anode (mol frac)': 'CO2RRCA',
    'CO conversion': 'CO2RRConv',  # Will be skipped (is in inputs)
    'Crossover rate': 'CO2RRCross',  # Will be skipped (is in inputs)
    'Unreacted (conversion)': 'CO2RRUR',
    'Energy consumption (MJ)': 'CO2RRE',
    'Regeneration energy (MJ) (Cathode)': 'CO2RRRC',
    'Regeneration energy (MJ) (Anode)': 'CO2RRRA',
    'Electrolyzer energy consumption (MJ)': 'CO2RREE',
    'Current (A)': 'CO2RRC',
    'Cell Area (m2)': 'CO2RRA',
    'NPV (USD)': 'CO2RRNPV',
    'MSP (USD/kgCO)': 'CO2RRMSP',
    'LCA (kgCO2eq/kgCO)': 'CO2RRLCA'
}


# =================================================

def process_datasets():
    if not os.path.exists(os.path.join(root_dir, INPUT_FILE)):
        print(f"❌ Error: Input file '{INPUT_FILE}' not found.")
        return

    print(f"📖 Loading {INPUT_FILE}...")
    try:
        df = pd.read_excel(os.path.join(root_dir, INPUT_FILE), engine='openpyxl')
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return

    print(f"\n🚀 Starting batch processing...\n")

    generated_count = 0
    skipped_count = 0

    # 2. Loop through targets
    for target_col, filename in TARGET_MAPPING.items():

        # --- EXCLUSION LOGIC ---
        # If the target is strictly inside the input feature list, skip it.
        if target_col in FEATURE_COLS:
            print(f"   🚫 SKIPPED: {filename:<15} (Reason: Target '{target_col}' is already an input feature)")
            skipped_count += 1
            continue
        # -----------------------

        if target_col not in df.columns:
            print(f"   ⚠️ SKIPPED: {filename:<15} (Reason: Column '{target_col}' not found in Excel)")
            continue

        # Create DataFrame: Fixed Features + This Target
        cols_to_keep = FEATURE_COLS + [target_col]

        # Verify features exist before saving
        missing_feats = [c for c in cols_to_keep if c not in df.columns]
        if missing_feats:
            print(f"   ❌ ERROR: {filename:<15} (Missing columns: {missing_feats})")
            continue

        df_subset = df[cols_to_keep].copy()

        # Save
        OUTPUT_FILE = f"{filename}.csv"
        df_subset.to_csv(os.path.join(root_dir, OUTPUT_FILE), index=False)
        generated_count += 1

    print("-" * 40)
    print(f"🎉 Process Complete!")
    print(f"   - Files Created: {generated_count}")
    print(f"   - Targets Skipped: {skipped_count}")


if __name__ == "__main__":
    process_datasets()
import pandas as pd
import os

# ================= CONFIGURATION =================
# 1. File Paths
root_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = 'data_parsed.csv'  # Replace with your actual raw file name

# 2. Target Column Name
# Enter the exact name of the Y variable in your raw csv
TARGET_COL = 'CO2_emission'    # minimum_selling_price, CO2_emission
name_to_save = {
    'CO2_emission': 'CO2HE',
    'minimum_selling_price': 'CO2HP'
}
OUTPUT_FILE = f'{name_to_save[TARGET_COL]}x10v2.csv'  # This is the file you will use in the main code

EXCLUDE_COLS = ['x7', 'x9', 'x16']
# =================================================

def clean_dataset():
    if not os.path.exists(os.path.join(root_dir, INPUT_FILE)):
        print(f"❌ Error: Input file '{INPUT_FILE}' not found.")
        return

    print(f"📖 Loading {INPUT_FILE}...")
    df = pd.read_csv(os.path.join(root_dir, INPUT_FILE))
    print(f"   - Initial shape: {df.shape}")

    # 1. Filter rows where flag is 'success'
    if 'flag' in df.columns:
        # .str.strip() handles potential whitespace like "success "
        df = df[df['flag'].astype(str).str.strip() == 'success']
        print(f"   - Rows after filtering 'success': {len(df)}")
    else:
        print("   ⚠️ Warning: 'flag' column not found. Skipping row filtering.")

    # if 'x15' in df.columns:
    #     df = df[df['x15'].astype(float) < 1]
    #     print(f"   - Rows after filtering 'x15' at the bound: {len(df)}")
    # else:
    #     print("   ⚠️ Warning: 'x15' column not found. Skipping row filtering.")

    if 'x10' in df.columns:
        df = df[df['x10'].astype(float) > 1]
        print(f"   - Rows after filtering 'x10' at the bound: {len(df)}")
    else:
        print("   ⚠️ Warning: 'x10' column not found. Skipping row filtering.")

    # 2. Select Features (x1 - x16)
    all_feature_cols = [f'x{i}' for i in range(1, 17)]
    feature_cols = [col for col in all_feature_cols if col not in EXCLUDE_COLS]

    # Verify these columns exist
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"❌ Error: The following expected columns are missing: {missing}")
        return

    # 3. Construct Final DataFrame
    # Format: [x1, x2, ..., x16, target]
    if TARGET_COL not in df.columns:
        print(f"❌ Error: Target column '{TARGET_COL}' not found in dataset.")
        return

    final_cols = feature_cols + [TARGET_COL]
    df_clean = df[final_cols].copy()

    # 4. Save to new file
    df_clean.to_csv(os.path.join(root_dir, OUTPUT_FILE), index=False)

    print("\n✅ Success!")
    print(f"   - Saved cleaned dataset to: {OUTPUT_FILE}")
    print(f"   - Final columns: {list(df_clean.columns)}")
    print(f"   - You can now load this file in your main KAN script.")


if __name__ == "__main__":
    clean_dataset()
import json
import re
from pathlib import Path

import pandas as pd

# ----------------------------
# Project Paths
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
CLEANED_DATA = PROJECT_ROOT / "data" / "cleaned"

INPUT_FILE = RAW_DATA / "all_systems_links.csv"
OUTPUT_FILE = CLEANED_DATA / "systems_links_clean.json"


# ----------------------------
# Load Data
# ----------------------------

def load_data():
    print("Loading systems links dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} records.")

    return df


# ----------------------------
# Clean Missing Values
# ----------------------------

def clean_missing_values(df):
    return df.fillna("")


# ----------------------------
# Clean Whitespace
# ----------------------------

def clean_whitespace(df):

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].astype(str).str.strip()

    return df


# ----------------------------
# Remove Duplicates
# ----------------------------

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"Removed {before-after} duplicate rows.")

    return df


# ----------------------------
# Rename Columns
# ----------------------------

def rename_columns(df):

    df = df.rename(columns={
        "System Source": "system_source",
        "Search Name": "search_name",
        "Detail URL": "detail_url"
    })

    return df


# ----------------------------
# Extract Plant ID
# ----------------------------

def extract_plant_id(df):

    df["plant_id"] = df["detail_url"].apply(
        lambda url: re.search(r"xplant_id/([A-Za-z0-9]+)", url).group(1)
        if url else ""
    )

    return df


# ----------------------------
# Validation
# ----------------------------

def validate_data(df):

    print("\n========== SYSTEM LINKS REPORT ==========")

    print(f"Total Records         : {len(df)}")
    print(f"Missing Search Names  : {(df['search_name'] == '').sum()}")
    print(f"Missing System Source : {(df['system_source'] == '').sum()}")
    print(f"Missing URLs          : {(df['detail_url'] == '').sum()}")

    print("-----------------------------------------")


# ----------------------------
# Save JSON
# ----------------------------

def save_json(df):

    records = df.to_dict(orient="records")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

    print(f"\nSaved cleaned data to:\n{OUTPUT_FILE}")


# ----------------------------
# Main
# ----------------------------

def main():

    df = load_data()

    df = clean_missing_values(df)

    df = clean_whitespace(df)

    df = remove_duplicates(df)

    df = rename_columns(df)

    df = extract_plant_id(df)

    validate_data(df)

    save_json(df)


if __name__ == "__main__":
    main()
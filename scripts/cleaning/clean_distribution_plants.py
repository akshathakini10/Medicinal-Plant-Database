import json
import re 
from pathlib import Path 

import pandas as pd

#Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
CLEANED_DATA = PROJECT_ROOT / "data" / "cleaned"

INPUT_FILE = RAW_DATA / "distribution_maps.csv"
OUTPUT_FILE = CLEANED_DATA / "distribution_clean.json"

#Load Data
def load_data():
    print("Loading distribution dataset..")
    df = pd.read_csv(INPUT_FILE) 
    print(f"Loaded {len(df)} records.")

    return df

#Clean Missing Values
def clean_missing_values(df):
    return df.fillna("")

#Clean Whitespace
def clean_whitespace(df):
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].astype(str).str.strip()
    return df

#Remove Duplicates
def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicates rows.")
    return df


#Rename Columns
def rename_columns(df):
    df = df.rename(columns={
        "Plant Name": "plant_name",
        "Distribution Map URL": "distribution_map_url"
    })

    return df

#Extract mapId
def extract_map_id(df):
    df["map_id"] = df["distribution_map_url"].apply(
        lambda url: re.search(r"/(\d+)\.jpg$", url).group(1)
        if url else ""
    )

    return df

#Validation
def validate_data(df):

    print("\n========== DISTRIBUTION REPORT ==========")

    print(f"Total Records        : {len(df)}")
    print(f"Missing Plant Names  : {(df['plant_name'] == '').sum()}")
    print(f"Missing URLs         : {(df['distribution_map_url'] == '').sum()}")

    print("-----------------------------------------")

#Save JSON
def save_json(df):

    records = df.to_dict(orient="records")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

    print(f"\nSaved cleaned data to:\n{OUTPUT_FILE}")

#Main
def main():

    df = load_data()

    df = clean_missing_values(df)

    df = clean_whitespace(df)

    df = remove_duplicates(df)

    df = rename_columns(df)

    df = extract_map_id(df)

    validate_data(df)

    save_json(df)


if __name__ == "__main__":
    main()
import json 
from pathlib import Path 

import pandas as pd 

#Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
CLEANED_DATA = PROJECT_ROOT / "data" / "cleaned"

INPUT_FILE = RAW_DATA / "final_medicinal_plants.csv"
OUTPUT_FILE = CLEANED_DATA / "medicinal_plants_clean.json"

#Load data
def load_data():
    """Load the Medicinal plats dataset."""
    print("Loading medicinal plats dataset..")
    df=pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records.")
    return df

#Cleaning Missing Values
def clean_missing_values(df):
    """Replace NaN values with empty strings."""

    return df.fillna("")

#Clean Whitespace
def clean_whitespace(df):
    """Remove leading and trailing spaces from all text columns."""

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].astype(str).str.strip()
    
    return df

#Clean Family 
def clean_family(df):
    """Convert family names to title case."""

    df["Family"]= df["Family"].str.title()

    return df

#Clean Systems
def clean_systems(df):
    """Convert systems into a list."""

    df["systems"] = df["Systems"].apply(
        lambda x: [item.strip() for item in x.split(",")] if x else []
    )

    return df

#Convert Synonyms
def clean_synonyms(df):
    """Convert synonym column into a list."""

    df["synonyms"] = df["Synonym"].apply(
        lambda x: [x] if x else []
    )

    return df

#Save JSON
def save_json(df):
    """Save cleaned data as JSON."""

    records = df.to_dict(orient="records")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

    print(f"\nSaved cleaned data to:\n{OUTPUT_FILE}")

#Validation 
def validate_data(df):
    """Print validation summary."""

    print("\n========== CLEANING REPORT ==========")

    print(f"Total Plants           : {len(df)}")
    print(f"Missing Family         : {(df['family'] == '').sum()}")
    print(f"Missing Systems        : {(df['systems'].apply(len) == 0).sum()}")
    print(f"Missing Synonyms       : {(df['synonyms'].apply(len) == 0).sum()}")
    print(f"Missing Vernacular     : {(df['vernacular_names'] == '').sum()}")

    print("---Done---")

#Rename the columns
def rename_columns(df):
    df = df.rename(columns={
        "Botanical Name": "botanical_name",
        "Family": "family",
        "Systems": "systems_raw",
        "Vernacular Names": "vernacular_names",
        "Detail URL": "detail_url"
    })

    return df

#Add plant ID
import re

def extract_plant_id(df):
    """Extract unique plant ID from Detail URL."""

    df["plant_id"] = df["detail_url"].apply(
        lambda url: re.search(r"xplant_id/([A-Za-z0-9]+)", url).group(1)
        if url else ""
    )

    return df


#Main Function 

def main():
    df = load_data()

    df = clean_missing_values(df)

    df = clean_whitespace(df)

    df = clean_family(df)

    df = clean_systems(df)

    df = clean_synonyms(df)

    df = rename_columns(df)

    df = extract_plant_id(df)

    validate_data(df)

    save_json(df)


if __name__ == "__main__":
    main()


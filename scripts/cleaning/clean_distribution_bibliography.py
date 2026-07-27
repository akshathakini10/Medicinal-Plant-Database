import os
import json
import pandas as pd

# ============================================
# FILE PATHS
# ============================================

INPUT_FILE = "data/raw/distribution_bibliography.csv"
OUTPUT_FILE = "data/cleaned/distribution_bibliography_clean.json"

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv(INPUT_FILE)

print("========== CLEANING REPORT ==========")
print(f"Rows before cleaning : {len(df)}")

# ============================================
# REMOVE DUPLICATES
# ============================================

df.drop_duplicates(
    subset=["Botanical Name", "Title", "Author"],
    inplace=True
)

print(f"Rows after removing duplicates : {len(df)}")

# ============================================
# HANDLE MISSING VALUES
# ============================================

df.fillna("", inplace=True)

# ============================================
# CLEAN TEXT
# ============================================

text_columns = [
    "Botanical Name",
    "Title",
    "Author",
    "Detail URL"
]

for col in text_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

# ============================================
# RENAME COLUMNS
# ============================================

df.rename(columns={
    "Botanical Name": "botanical_name",
    "Title": "title",
    "Author": "author",
    "Detail URL": "detail_url"
}, inplace=True)

# ============================================
# CREATE UNIQUE ID
# ============================================

df.insert(
    0,
    "reference_id",
    ["REF{:06d}".format(i + 1) for i in range(len(df))]
)

# ============================================
# SAVE JSON
# ============================================

os.makedirs("data/cleaned", exist_ok=True)

records = df.to_dict(orient="records")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=4, ensure_ascii=False)

# ============================================
# SUMMARY
# ============================================

print("\n========== FINAL REPORT ==========")
print("Total Records :", len(df))
print("Columns :", list(df.columns))
print("Saved to :", OUTPUT_FILE)
print("==================================")
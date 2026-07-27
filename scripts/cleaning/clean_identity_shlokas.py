import pandas as pd
import re

# ===========================================
# FUNCTION TO CLEAN TEXT
# ===========================================

def clean_text(value):
    """
    Cleans text by:
    - Keeping empty values as None
    - Removing leading/trailing spaces
    - Replacing multiple spaces with one
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    value = re.sub(r"\s+", " ", value)

    if value == "":
        return None

    return value


# ===========================================
# CLEAN AYURVEDIC IDENTITY DATASET
# ===========================================

print("="*60)
print("Cleaning Ayurvedic Identity Dataset")
print("="*60)

identity = pd.read_csv(
    "data/raw/ayurvedic_identity_full.csv",
    encoding="utf-8"
)

print("Original Shape:", identity.shape)

# Remove duplicate rows
duplicates = identity.duplicated().sum()
print("Duplicate Rows Found:", duplicates)

identity = identity.drop_duplicates()

# Clean every text column
for col in identity.columns:
    identity[col] = identity[col].apply(clean_text)

identity["status"] = identity["status"].where(identity["status"].notna(), None)
identity["discussion"] = identity["discussion"].where(identity["discussion"].notna(), None)

# Standardize column names
identity.columns = (
    identity.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nMissing Values:")
print(identity.isnull().sum())

# Save as JSON
identity.to_json(
    "data/cleaned/ayurvedic_identity_clean.json",
    orient="records",
    force_ascii=False,
    indent=4
)

print("\nSaved:")
print("data/cleaned/ayurvedic_identity_clean.json")


# ===========================================
# CLEAN SHLOKAS DATASET
# ===========================================

print("\n")
print("="*60)
print("Cleaning Shlokas Dataset")
print("="*60)

shlokas = pd.read_csv(
    "data/raw/shlokas_full.csv",
    encoding="utf-8"
)

print("Original Shape:", shlokas.shape)

duplicates = shlokas.duplicated().sum()

print("Duplicate Rows Found:", duplicates)

shlokas = shlokas.drop_duplicates()

# Clean every text column
for col in shlokas.columns:
    shlokas[col] = shlokas[col].apply(clean_text)

# Fill missing values
shlokas["image_urls"] = shlokas["image_urls"].where(shlokas["image_urls"].notna(), None)
shlokas["ocr_text"] = shlokas["ocr_text"].where(shlokas["ocr_text"].notna(), None)
# Standardize column names
shlokas.columns = (
    shlokas.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nMissing Values:")
print(shlokas.isnull().sum())

# Save as JSON
shlokas.to_json(
    "data/cleaned/shlokas_clean.json",
    orient="records",
    force_ascii=False,
    indent=4
)

print("\nSaved:")
print("data/cleaned/shlokas_clean.json")

print("\n")
print("="*60)
print("CLEANING COMPLETED SUCCESSFULLY")
print("="*60)
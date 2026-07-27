import json


def check_json(file_path, dataset_name):

    print("=" * 70)
    print(dataset_name)
    print("=" * 70)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total Records: {len(data)}")

    # -----------------------------
    # Check keys
    # -----------------------------
    print("\nColumns:")

    for key in data[0].keys():
        print("✓", key)

    # -----------------------------
    # Check Unicode
    # -----------------------------
    unicode_found = False

    for record in data:
        for value in record.values():

            if isinstance(value, str):

                if any('\u0900' <= c <= '\u097F' for c in value):

                    print("\nSample Sanskrit:")
                    print(value[:100])

                    unicode_found = True
                    break

        if unicode_found:
            break

    if not unicode_found:
        print("\nNo Sanskrit text found.")

    # -----------------------------
    # Count null values
    # -----------------------------

    null_count = 0

    for record in data:

        for value in record.values():

            if value is None:

                null_count += 1

    print("\nTotal null values:", null_count)

    # -----------------------------
    # Check for Not Available
    # -----------------------------

    not_available = 0

    for record in data:

        for value in record.values():

            if value == "Not Available":

                not_available += 1

    print("Occurrences of 'Not Available':", not_available)

    if not_available == 0:

        print("\n✅ PASS")

    else:

        print("\n❌ Replace remaining 'Not Available' values.")


check_json(
    "data/cleaned/ayurvedic_identity_clean.json",
    "AYURVEDIC IDENTITY"
)

print()

check_json(
    "data/cleaned/shlokas_clean.json",
    "SHLOKAS"
)
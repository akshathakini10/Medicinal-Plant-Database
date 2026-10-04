import json

FILE = "data/vector_store/metadata.json"

with open(FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

names = ["neem", "tulsi", "ashoka", "citraka"]

for search_name in names:

    print("\n" + "=" * 60)
    print("SEARCH:", search_name)
    print("=" * 60)

    found = 0

    for item in metadata:

        text = json.dumps(
            item,
            ensure_ascii=False
        ).lower()

        if search_name in text:

            print(json.dumps(
                item,
                ensure_ascii=False,
                indent=2
            ))

            found += 1

            if found >= 5:
                break

    print("Matches shown:", found)
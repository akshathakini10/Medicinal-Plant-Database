from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
import time

# ---------- START BROWSER ----------

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install())
)

driver.get("https://www.medicinalplants.in/sanskritauthentication")

print("Opening website...")
time.sleep(5)

iframe = driver.find_element("tag name", "iframe")
driver.switch_to.frame(iframe)

print("Entered iframe")

# ---------- SCRAPING ----------

records = []

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

for letter in letters:

    print(f"\n========== LETTER {letter} ==========")

    try:
        driver.find_element(
            "id",
            f"checksanskritcit{letter}"
        ).click()

        time.sleep(2)

    except Exception as e:
        print("Could not open letter", letter)
        continue

    page_no = 1

    while True:

        print(f"Scraping {letter} Page {page_no}")

        soup = BeautifulSoup(
            driver.page_source,
            "html.parser"
        )

        briefs = soup.find_all(
            "div",
            id=lambda x: x and x.startswith("brief_")
        )

        print("Drugs on page:", len(briefs))

        for brief in briefs:

            idx = brief["id"].replace("brief_", "")

            detail = soup.find(
                "div",
                id=f"details_{idx}"
            )

            if not detail:
                continue

            english = brief.find(
                "div",
                class_="citationmast_english"
            )

            sanskrit = brief.find(
                "div",
                class_="citationmast_devanagari"
            )

            translit = brief.find(
                "div",
                class_="citationmast_diacritical"
            )

            english = (
                english.get_text(strip=True)
                if english else ""
            )

            sanskrit = (
                sanskrit.get_text(strip=True)
                if sanskrit else ""
            )

            translit = (
                translit.get_text(strip=True)
                if translit else ""
            )

            table = detail.find("table")

            if not table:
                continue

            rows = table.find_all("tr")[1:]

            for row in rows:

                cols = row.find_all("td")

                if len(cols) < 5:
                    continue

                records.append({
                    "letter": letter,
                    "page": page_no,

                    "drug_name": english,
                    "sanskrit_name": sanskrit,
                    "transliteration": translit,

                    "ref_drug_name":
                        cols[0].get_text(
                            " ",
                            strip=True
                        ),

                    "botanical_correlation":
                        cols[1].get_text(
                            " ",
                            strip=True
                        ),

                    "status":
                        cols[2].get_text(
                            " ",
                            strip=True
                        ),

                    "discussion":
                        cols[3].get_text(
                            " ",
                            strip=True
                        ),

                    "reference":
                        cols[4].get_text(
                            " ",
                            strip=True
                        )
                })

        # ---------- NEXT PAGE ----------

        try:

            old_html = driver.page_source

            driver.find_element(
                "id",
                "srinipaginationNext"
            ).click()

            time.sleep(2)

            new_html = driver.page_source

            if new_html == old_html:
                break

            page_no += 1

        except:
            break

# ---------- SAVE ----------

driver.quit()

with open(
    "data/ayurvedic_identity_full.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        records,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n===================================")
print("TOTAL CORRELATION RECORDS:", len(records))
print("Saved to data/ayurvedic_identity_full.json")
print("===================================")
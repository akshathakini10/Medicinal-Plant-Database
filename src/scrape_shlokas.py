from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
import time

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install())
)

driver.get("https://www.medicinalplants.in/sanskritappnuse")

time.sleep(5)

iframe = driver.find_element("tag name", "iframe")
driver.switch_to.frame(iframe)

records = []

page_no = 1

while True:

    print(f"\nScraping Page {page_no}")

    soup = BeautifulSoup(driver.page_source, "html.parser")

    briefs = soup.find_all(
        "div",
        id=lambda x: x and x.startswith("brief_")
    )

    print("Entries:", len(briefs))

    for brief in briefs:

        idx = brief["id"].replace("brief_", "")

        try:
            driver.find_element("id", f"brief_{idx}").click()
            time.sleep(1)
        except:
            continue

        page_soup = BeautifulSoup(
            driver.page_source,
            "html.parser"
        )

        detail = page_soup.find(
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

        imgs = detail.find_all("img")

        image_urls = []

        for img in imgs:

            src = img.get("src")

            if src:
                image_urls.append(src)

        records.append({
            "page": page_no,
            "drug_name": english,
            "sanskrit_name": sanskrit,
            "transliteration": translit,
            "image_urls": image_urls
        })

    # NEXT PAGE

    try:

        old_html = driver.page_source

        driver.find_element(
            "id",
            "srinipaginationNext"
        ).click()

        time.sleep(3)

        if driver.page_source == old_html:
            break

        page_no += 1

    except:
        break

driver.quit()

with open(
    "data/shlokas_images.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        records,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n================================")
print("Saved records:", len(records))
print("data/shlokas_images.json")
print("================================")
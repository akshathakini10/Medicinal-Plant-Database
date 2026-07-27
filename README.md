
# Medicinal Plant Database Scraper

## Overview

This project focuses on collecting, organizing, and cleaning information from the **Medicinal Plants Database** maintained by the **Foundation for Revitalisation of Local Health Traditions (FRLHT)**.

The objective of this project is to create a structured, cleaned, and searchable dataset containing information about medicinal plants used across different traditional systems of medicine in India. These cleaned datasets will serve as the foundation for a future **Multimodal Retrieval-Augmented Generation (RAG)** system for Ayurvedic knowledge retrieval.

---

# Data Collected

## 1. Medicinal Plant Database

A total of **7263 medicinal plant records** were collected.

The dataset contains:

* Botanical Name
* Family
* Systems of Medicine
* Synonyms
* Vernacular Names
* Detail URL

**Raw Dataset**

```text
data/raw/final_medicinal_plants.csv
```

**Cleaned Dataset**

```text
data/cleaned/medicinal_plants_clean.json
```

---

## 2. Ayurvedic Identity

A total of **1407 Ayurvedic drug names** and approximately **3900 botanical correlation records** were collected.

The dataset contains:

* Drug Name
* Sanskrit Name
* Transliteration
* Botanical Correlation
* Correlation Status
* Discussion
* References

**Raw Dataset**

```text
data/raw/ayurvedic_identity_full.csv
```

**Cleaned Dataset**

```text
data/cleaned/ayurvedic_identity_clean.json
```

---

## 3. Sanskrit Application and Use (Shlokas)

A total of **290 medicinal plant entries** and **917 OCR-processed shloka images** were collected.

The dataset contains:

* Drug Name
* Sanskrit Name
* Transliteration
* Sanskrit Shlokas
* English Explanation
* References

**Raw Dataset**

```text
data/raw/shlokas_full.csv
```

**Cleaned Dataset**

```text
data/cleaned/shlokas_clean.json
```

---

## 4. Distribution Maps

A total of **1101 distribution map records** were collected.

The dataset contains:

* Plant Name
* Distribution Map URL

**Raw Dataset**

```text
data/raw/distribution_maps.csv
```

**Cleaned Dataset**

```text
data/cleaned/distribution_clean.json
```

---

## 5. Distribution Bibliography

Bibliographic references associated with the geographical distribution of medicinal plants were collected from the FRLHT database.

The dataset contains:

* Botanical Name
* Bibliography Title
* Author(s)
* Detail URL

**Raw Dataset**

```text
data/raw/distribution_bibliography.csv
```

**Cleaned Dataset**

```text
data/cleaned/distribution_bibliography_clean.json
```

---

## 6. Statewise Medicinal Plant Lists

Medicinal plant checklists were collected for the following states:

* Karnataka
* Kerala
* Tamil Nadu
* Andhra Pradesh
* Chhattisgarh
* Odisha
* Sikkim
* Rajasthan
* West Bengal
* Maharashtra

Files:

```text
data/statewise_pdfs/
```

---

## 7. Trade Information

Trade-related information was collected from official documents available on the website.

Files:

```text
data/trade_pdfs/
data/trade_text/
```

The extracted information includes:

* Botanical Name
* Trade Name
* Sanskrit Names
* Family Name
* Parts Used
* Source
* Medical Systems
* Annual Trade Volume
* Price Range

---

## 8. Static Website Content

The following sections were archived:

* FAQ
* About Database
* About FRLHT
* About NMPB
* Contact
* Team

Files:

```text
data/static_pages/
```

---

# Project Structure

```text
Medicinal-Plant-Database
│
├── data
│   ├── raw
│   │   ├── final_medicinal_plants.csv
│   │   ├── ayurvedic_identity_full.csv
│   │   ├── shlokas_full.csv
│   │   ├── distribution_maps.csv
│   │   ├── distribution_bibliography.csv
│   │   └── all_systems_links.csv
│   │
│   ├── cleaned
│   │   ├── medicinal_plants_clean.json
│   │   ├── ayurvedic_identity_clean.json
│   │   ├── shlokas_clean.json
│   │   ├── systems_links_clean.json
│   │   ├── distribution_clean.json
│   │   └── distribution_bibliography_clean.json
│   │
│   ├── statewise_pdfs
│   ├── trade_pdfs
│   ├── trade_text
│   └── static_pages
│
├── scripts
│   ├── scraping
│   └── cleaning
│       ├── clean_medicinal_plants.py
│       ├── clean_systems_links.py
│       ├── clean_distribution.py
│       ├── clean_distribution_bibliography.py
│       ├── clean_identity_shlokas.py
│       └── check_cleaned_json.py
│
├── src
├── tests
├── .gitignore
└── README.md
```

---

# Data Cleaning

The collected datasets were cleaned and standardized before being converted into JSON format.

Cleaning operations performed include:

* Removed duplicate records
* Trimmed leading and trailing whitespace
* Normalized multiple spaces
* Standardized column names
* Preserved Sanskrit Unicode characters
* Standardized missing values
* Generated structured JSON datasets
* Validated cleaned JSON files

The cleaned datasets are available in:

```text
data/cleaned/
├── medicinal_plants_clean.json
├── ayurvedic_identity_clean.json
├── shlokas_clean.json
├── systems_links_clean.json
├── distribution_clean.json
└── distribution_bibliography_clean.json
```

---

# Technologies Used

* Python
* Requests
* BeautifulSoup
* Selenium
* Pandas
* PyMuPDF
* OCR Processing
* JSON
* Git
* GitHub

---

# Contributors

* Akshatha Kini
* Ananya Shetty

---

# Disclaimer

This project was created for educational and research purposes. All data belongs to the original source website and the respective organizations maintaining it. The datasets are provided solely for academic study, analysis, and learning purposes.

---

### A couple of small suggestions

* If you rename `distribution_clean.json` to `distribution_maps_clean.json`, update the README accordingly so it's consistent with `distribution_bibliography_clean.json`.
* In the project structure, list your actual cleaning script names (for example, `clean_distribution_bibliography.py`) exactly as they appear in the repository. This avoids confusion for anyone trying to run them.

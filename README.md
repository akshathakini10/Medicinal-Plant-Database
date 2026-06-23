# Medicinal Plant Database Scraper

## Overview

This project focuses on collecting and organizing information from the Medicinal Plants Database maintained by FRLHT (Foundation for Revitalisation of Local Health Traditions).

The objective of this project is to create a structured and searchable dataset containing information about medicinal plants used across different traditional systems of medicine in India.

---

## Data Collected

### 1. Medicinal Plant Database

A total of **7263 medicinal plant records** were collected.

The dataset contains:

* Botanical Name
* Family
* Systems of Medicine
* Synonyms
* Vernacular Names
* Detail URLs

Dataset:

```text
data/final_medicinal_plants.csv
```

---

### 2. Ayurvedic Identity

A total of **1407 Ayurvedic drug names** and approximately **3900 botanical correlation records** were collected.

The dataset contains:

* Drug Name
* Sanskrit Name
* Transliteration
* Botanical Correlation
* Correlation Status
* Discussion
* References

Dataset:

```text
data/ayurvedic_identity_full.csv
```

---

### 3. Sanskrit Application and Use (Shlokas)

A total of **290 medicinal plant entries** and **917 OCR-processed shloka images** were collected.

The dataset contains:

* Drug Name
* Sanskrit Name
* Transliteration
* Sanskrit Shlokas
* English Explanations
* References

Dataset:

```text
data/shlokas_full.csv
```

---

### 4. Distribution Maps

A total of **1101 distribution map records** were collected.

The dataset contains:

* Plant Name
* Distribution Map URL

Dataset:

```text
data/distribution_maps.csv
```

---

### 5. Statewise Medicinal Plant Lists

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

### 6. Trade Information

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

### 7. Static Website Content

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

## Project Structure

```text
MedicinalPlantScraper
│
├── data
│   ├── final_medicinal_plants.csv
│   ├── ayurvedic_identity_full.csv
│   ├── shlokas_full.csv
│   ├── distribution_maps.csv
│   ├── statewise_pdfs
│   ├── trade_pdfs
│   ├── trade_text
│   └── static_pages
│
├── src
│   ├── scrape_all_system_links.py
│   ├── scrape_final_dataset.py
│   ├── scrape_distribution_maps.py
│   ├── scrape_ayurvedic_identity.py
│   ├── scrape_shlokas.py
│   ├── scrape_static_pages.py
│   └── supporting scripts
│
└── README.md
```

---

## Technologies Used

* Python
* Requests
* BeautifulSoup
* Pandas
* PyMuPDF
* OCR Processing

---

## Contributors

* Akshatha Kini
* Ananya Shetty

---

## Disclaimer

This project was created for educational and research purposes. All data belongs to the original source website and the respective organizations maintaining it. The datasets are provided solely for academic study, analysis, and learning purposes.

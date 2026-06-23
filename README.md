# Medicinal Plant Database Scraper

## Overview

This project focuses on collecting and organizing information from the Medicinal Plants Database maintained by FRLHT (Foundation for Revitalisation of Local Health Traditions).

The goal of the project is to create a structured dataset containing information about medicinal plants used in different traditional systems of medicine in India.

## Data Collected

### Medicinal Plant Database

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

### Distribution Maps

A total of **1101 distribution map records** were collected.

The dataset contains:

* Plant Name
* Distribution Map URL

Dataset:

```text
data/distribution_maps.csv
```

### Trade Information

Trade information was collected from the documents available on the website.

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

### Static Website Content

The following pages were archived:

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

## Project Structure

```text
MedicinalPlantScraper
│
├── data
│   ├── final_medicinal_plants.csv
│   ├── distribution_maps.csv
│   ├── trade_pdfs
│   ├── trade_text
│   └── static_pages
│
├── src
│   ├── scrape_all_system_links.py
│   ├── scrape_final_dataset.py
│   ├── scrape_distribution_maps.py
│   ├── scrape_static_pages.py
│   ├── download_trade_pdfs.py
│   └── extract_trade_pdf.py
│
└── README.md
```

## Technologies Used

* Python
* Requests
* BeautifulSoup
* Pandas
* PyMuPDF

## Contributors

* Akshatha Kini
* Ananya Shetty

## Disclaimer

This project was created for educational and research purposes. All data belongs to the original source website and the respective organizations maintaining it.

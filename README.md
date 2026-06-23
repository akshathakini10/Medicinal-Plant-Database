# Medicinal Plant Database Scraper

This project was developed to collect and organize data from the Medicinal Plants Database maintained by FRLHT (Foundation for Revitalisation of Local Health Traditions).

The objective of the project is to create a structured dataset containing information about medicinal plants used across different traditional systems of medicine in India.

## Data Collected

### 1. Medicinal Plant Records

The project contains information for 7263 medicinal plant entries including:

* Botanical Name
* Family
* Systems of Medicine
* Synonyms
* Vernacular Names

Dataset:

```text
data/final_medicinal_plants.csv
```

### 2. Distribution Maps

Distribution map information was collected for 1101 medicinal plants.

Dataset:

```text
data/distribution_maps.csv
```

### 3. Trade Information

Trade-related information was collected from the official trade documents available on the website.

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

### 4. Website Information Pages

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


## Notes

The project was created for educational and research purposes. All information belongs to the original source website and respective organizations.

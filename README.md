# Medicinal Plant Database Scraper

## About the Project

This project was developed to collect medicinal plant information from the Indian Medicinal Plants Database website.

The website contains information about medicinal plants used in different traditional systems of medicine such as Ayurveda, Siddha, Unani, Homeopathy, Folk Medicine and Sowa-Rigpa. The data was scraped and organized into a structured CSV dataset for further analysis and future application development.

## Dataset Information

The final dataset contains:

* 7263 medicinal plant records
* 182+ plant families
* Botanical names
* Plant families
* Systems of medicine
* Botanical synonyms
* Vernacular names
* Detail page URLs

## How the Data Was Collected

The website's search functionality was analyzed to identify the APIs used internally.

The scraper then:

1. Collected plant links from all six medical systems.
2. Removed duplicate links.
3. Visited each plant detail page.
4. Extracted useful information.
5. Stored the results in CSV format.

## Technologies Used

* Python
* Requests
* BeautifulSoup
* Pandas
* Git & GitHub

## Files

* `all_systems_links.csv` – All collected plant links
* `final_medicinal_plants.csv` – Final cleaned dataset
* `scrape_all_system_links.py` – Collects plant links
* `scrape_final_dataset.py` – Extracts plant details
* `readcsv.py` – Dataset inspection script

## Future Work

Some possible improvements for this project:

* Extract plant images
* Build a web application for searching plants
* Create a medicinal plant recommendation system
* Use the dataset for machine learning projects

## Contributors

* Akshatha Kini
* Ananya Shetty

## Note

This project was created for educational and research purposes. The original plant information belongs to the Indian Medicinal Plants Database and its respective maintainers.

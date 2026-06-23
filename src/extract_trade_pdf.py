import fitz  # PyMuPDF
import pandas as pd
import os

pdf_folder = "../data/trade_pdfs"
output_folder = "../data/trade_text"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, file)
        print("Reading:", file)

        doc = fitz.open(pdf_path)
        all_text = ""

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            all_text += f"\n\n--- Page {page_num} ---\n{text}"

        output_file = os.path.join(output_folder, file.replace(".pdf", ".txt"))

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(all_text)

        print("Saved:", output_file)
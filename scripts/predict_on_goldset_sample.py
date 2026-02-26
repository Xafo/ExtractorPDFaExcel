import csv
import os
import sys

import fitz


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import archivos


IN_CSV = "./goldset/goldset_manual_v1.csv"
OUT_CSV = "./goldset/goldset_predictions_v1.csv"
PDF_DIR = "./pdfs"


def main():
    with open(IN_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    by_pdf = {}
    for r in rows:
        by_pdf.setdefault(r["pdf_filename"], []).append(r)

    for pdf_name, items in by_pdf.items():
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"Skip missing PDF: {pdf_path}")
            continue
        print(f"Predict {pdf_name} ({len(items)} samples)")
        with fitz.open(pdf_path) as doc:
            for item in items:
                idx = int(item["page_idx_0based"])
                pred = archivos.extract_page(doc, idx)
                row = {
                    "sample_id": item["sample_id"],
                    "pdf_filename": item["pdf_filename"],
                    "page_num_1based": item["page_num_1based"],
                }
                for f in archivos.FIELDS:
                    row[f] = str(pred.get(f, "") or "")
                out_rows.append(row)

    fieldnames = ["sample_id", "pdf_filename", "page_num_1based"] + archivos.FIELDS
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Predictions -> {OUT_CSV} | rows={len(out_rows)}")


if __name__ == "__main__":
    main()

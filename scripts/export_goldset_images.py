import csv
import os

import fitz


GOLDSET_CSV = "./goldset/goldset_manual_v1.csv"
PDF_DIR = "./pdfs"
OUT_DIR = "./goldset/pages"
ZOOM = 2.2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(GOLDSET_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    by_pdf = {}
    for r in rows:
        by_pdf.setdefault(r["pdf_filename"], []).append(r)

    written = 0
    for pdf_name, items in by_pdf.items():
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"Skip missing PDF: {pdf_path}")
            continue

        with fitz.open(pdf_path) as doc:
            for item in items:
                idx = int(item["page_idx_0based"])
                if idx < 0 or idx >= len(doc):
                    continue
                page = doc.load_page(idx)
                pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), alpha=False)
                out_name = f"{item['sample_id']}.png"
                out_path = os.path.join(OUT_DIR, out_name)
                pix.save(out_path)
                written += 1

    print(f"Exported images: {written} -> {OUT_DIR}")


if __name__ == "__main__":
    main()

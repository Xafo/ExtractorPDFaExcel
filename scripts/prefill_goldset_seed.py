import csv
import os
import sys
import argparse

import fitz


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import archivos


IN_CSV = "./goldset/goldset_manual_v1.csv"
OUT_CSV = "./goldset/goldset_manual_v1_seed.csv"
PDF_DIR = "./pdfs"


def write_rows(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=IN_CSV)
    ap.add_argument("--output", default=OUT_CSV)
    ap.add_argument("--only-pdf", default="", help="Process only this PDF filename")
    ap.add_argument("--max-pdfs", type=int, default=0, help="Optional max number of PDFs to process")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    by_pdf = {}
    for r in rows:
        by_pdf.setdefault(r["pdf_filename"], []).append(r)

    processed = 0
    for pdf_name in sorted(by_pdf.keys()):
        if args.only_pdf and pdf_name != args.only_pdf:
            continue
        if args.max_pdfs and processed >= args.max_pdfs:
            break

        items = by_pdf[pdf_name]
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"Skip missing PDF: {pdf_path}")
            continue
        print(f"Prefill {pdf_name} ({len(items)} samples)")
        with fitz.open(pdf_path) as doc:
            for item in items:
                idx = int(item["page_idx_0based"])
                pred = archivos.extract_page(doc, idx)
                for f in archivos.FIELDS:
                    item[f] = str(pred.get(f, "") or "")
                item["status"] = "in_progress"
                item["notes"] = "seed_from_model"
        processed += 1

        if fieldnames:
            write_rows(args.output, rows, fieldnames)
            print(f"Checkpoint write -> {args.output}")

    if not fieldnames:
        print("No rows found.")
        return

    write_rows(args.output, rows, fieldnames)

    print(f"Seed file -> {args.output}")


if __name__ == "__main__":
    main()

import csv
import glob
import os
from math import floor

import fitz


PDF_DIR = "./pdfs"
OUT_CSV = "./goldset/goldset_manual_v1.csv"

FIELDS = [
    "certificado_no",
    "marca",
    "chasis",
    "linea",
    "pasajeros",
    "modelo",
    "clase",
    "placa",
    "motor",
    "color",
]


def round_half_up(x: float) -> int:
    return floor(x + 0.5)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def select_pages_1based(n_pages: int):
    if n_pages >= 20:
        raw = [
            1,
            2,
            round_half_up(0.28 * n_pages),
            round_half_up(0.50 * n_pages),
            round_half_up(0.56 * n_pages),
            round_half_up(0.75 * n_pages),
            n_pages - 1,
            n_pages,
        ]
        target = 8
    else:
        raw = [
            1,
            2,
            round_half_up(0.33 * n_pages),
            round_half_up(0.58 * n_pages),
            n_pages - 1,
            n_pages,
        ]
        target = 6

    pages = sorted(set(clamp(p, 1, n_pages) for p in raw))

    if len(pages) < target:
        mid = round_half_up(0.50 * n_pages)
        q1 = round_half_up(0.25 * n_pages)
        q3 = round_half_up(0.75 * n_pages)
        candidates = [p for p in range(1, n_pages + 1) if p not in pages]
        candidates.sort(key=lambda p: (min(abs(p - mid), abs(p - q1), abs(p - q3)), p))
        for p in candidates:
            pages.append(p)
            if len(pages) == target:
                break
        pages.sort()

    return pages


def selection_rule(page, n_pages):
    checkpoints = {
        1: "fixed:1",
        2: "fixed:2",
        n_pages - 1: "fixed:n-1",
        n_pages: "fixed:n",
        round_half_up(0.28 * n_pages): "anchor:0.28n",
        round_half_up(0.50 * n_pages): "anchor:0.50n",
        round_half_up(0.56 * n_pages): "anchor:0.56n",
        round_half_up(0.75 * n_pages): "anchor:0.75n",
        round_half_up(0.33 * n_pages): "anchor:0.33n",
        round_half_up(0.58 * n_pages): "anchor:0.58n",
    }
    return checkpoints.get(page, "fill:deterministic")


def stratum(page, n_pages):
    rel = page / float(max(1, n_pages))
    if page in (1, 2):
        return "hard_front"
    if page in (n_pages - 1, n_pages):
        return "hard_end"
    if rel <= 0.35:
        return "begin"
    if rel >= 0.70:
        return "end"
    return "middle"


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {PDF_DIR}")

    rows = []
    for pdf_path in pdfs:
        with fitz.open(pdf_path) as doc:
            n_pages = len(doc)
        picks = select_pages_1based(n_pages)
        quota = len(picks)
        pdf_name = os.path.basename(pdf_path)
        pdf_id = os.path.splitext(pdf_name)[0]

        for p in picks:
            row = {
                "sample_id": f"{pdf_id}_p{p:03d}",
                "pdf_id": pdf_id,
                "pdf_filename": pdf_name,
                "page_num_1based": p,
                "page_idx_0based": p - 1,
                "n_pages_in_pdf": n_pages,
                "quota_for_pdf": quota,
                "stratum": stratum(p, n_pages),
                "selection_rule": selection_rule(p, n_pages),
                "reviewer": "",
                "status": "todo",
                "quality_flag": "",
                "notes": "",
                "labeled_at_utc": "",
            }
            for f in FIELDS:
                row[f] = ""
            rows.append(row)

    cols = [
        "sample_id",
        "pdf_id",
        "pdf_filename",
        "page_num_1based",
        "page_idx_0based",
        "n_pages_in_pdf",
        "quota_for_pdf",
        "stratum",
        "selection_rule",
        "reviewer",
        "status",
    ] + FIELDS + [
        "quality_flag",
        "notes",
        "labeled_at_utc",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"Gold set scaffold -> {OUT_CSV}")
    print(f"Samples: {len(rows)} | PDFs: {len(pdfs)}")


if __name__ == "__main__":
    main()

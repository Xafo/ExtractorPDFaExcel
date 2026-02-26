import csv
import os
from math import ceil


IN_CSV = "./goldset/goldset_manual_v1_seed.csv"
OUT_DIR = "./goldset/batches"
BATCH_SIZE = 20

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

NOISE_TOKENS = {
    "CHASIS",
    "LINEA",
    "PASAJEROS",
    "MODELO",
    "CLASE",
    "PLACA",
    "MOTOR",
    "COLOR",
    "BENEFICIOS",
    "AMPARADOS",
    "COBERTURAS",
}


def risk_score(row):
    filled = sum(1 for f in FIELDS if str(row.get(f, "")).strip())
    score = (10 - filled) * 3

    marca = str(row.get("marca", "")).upper()
    chasis = str(row.get("chasis", "")).upper()
    color = str(row.get("color", "")).upper()

    if any(tok in marca for tok in NOISE_TOKENS):
        score += 6
    if any(tok in color for tok in NOISE_TOKENS):
        score += 4
    if len(chasis) not in (0, 17):
        score += 4
    if not row.get("placa", ""):
        score += 1
    if not row.get("motor", ""):
        score += 1

    # Prefer hard strata first for faster quality gains
    st = str(row.get("stratum", ""))
    if st.startswith("hard"):
        score += 2

    return score


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(IN_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []

    todo_rows = [r for r in rows if r.get("status", "").strip() != "qa_done"]
    todo_rows.sort(key=lambda r: (-risk_score(r), r.get("sample_id", "")))

    total = len(todo_rows)
    n_batches = ceil(total / float(BATCH_SIZE)) if total else 0

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        batch = todo_rows[start:end]
        path = os.path.join(OUT_DIR, f"annotation_batch_{i+1:02d}.csv")
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(batch)

    print(f"Input rows: {len(rows)} | Pending rows: {total}")
    print(f"Batches: {n_batches} | Size: {BATCH_SIZE}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()

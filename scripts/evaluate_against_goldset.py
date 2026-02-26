import argparse
import csv


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


def norm(x: str) -> str:
    return " ".join((x or "").upper().split())


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="./goldset/goldset_manual_v1.csv")
    ap.add_argument("--pred", default="./goldset/goldset_predictions_v1.csv")
    ap.add_argument("--require-status", default="qa_done", help="Only evaluate gold rows with this status")
    ap.add_argument("--out-mismatches", default="", help="Optional CSV path for per-field mismatches")
    args = ap.parse_args()

    gold_rows = load_csv(args.gold)
    pred_rows = load_csv(args.pred)
    pred_by_id = {r["sample_id"]: r for r in pred_rows}

    eval_rows = [r for r in gold_rows if (r.get("status", "") == args.require_status)]
    if not eval_rows:
        print(f"No rows with status='{args.require_status}'. Nothing to evaluate.")
        return

    total_cells = 0
    filled_cells = 0
    exact_cells = 0
    exact_rows = 0

    field_total = {f: 0 for f in FIELDS}
    field_fill = {f: 0 for f in FIELDS}
    field_exact = {f: 0 for f in FIELDS}
    mismatches = []

    for g in eval_rows:
        p = pred_by_id.get(g["sample_id"], {})
        row_ok = True
        for f in FIELDS:
            gv = norm(g.get(f, ""))
            pv = norm(p.get(f, ""))
            total_cells += 1
            field_total[f] += 1

            if pv:
                filled_cells += 1
                field_fill[f] += 1

            if pv == gv:
                exact_cells += 1
                field_exact[f] += 1
            else:
                row_ok = False
                mismatches.append(
                    {
                        "sample_id": g.get("sample_id", ""),
                        "pdf_filename": g.get("pdf_filename", ""),
                        "page_num_1based": g.get("page_num_1based", ""),
                        "field": f,
                        "gold": gv,
                        "pred": pv,
                    }
                )

        if row_ok:
            exact_rows += 1

    n_rows = len(eval_rows)
    print(f"Evaluated rows: {n_rows}")
    print(f"Fill Rate: {filled_cells/total_cells*100:.2f}%")
    print(f"Exact Cell Accuracy: {exact_cells/total_cells*100:.2f}%")
    print(f"Exact Row Accuracy (10/10): {exact_rows/n_rows*100:.2f}%")
    print("\nPer field:")
    for f in FIELDS:
        fill = field_fill[f] / field_total[f] * 100
        ex = field_exact[f] / field_total[f] * 100
        print(f"{f:15s} fill={fill:6.2f}% exact={ex:6.2f}%")

    if args.out_mismatches:
        with open(args.out_mismatches, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["sample_id", "pdf_filename", "page_num_1based", "field", "gold", "pred"],
            )
            w.writeheader()
            w.writerows(mismatches)
        print(f"\nMismatches -> {args.out_mismatches} ({len(mismatches)} rows)")


if __name__ == "__main__":
    main()

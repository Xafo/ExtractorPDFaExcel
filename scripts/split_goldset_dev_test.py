import csv
import hashlib
import os


IN_CSV = "./goldset/goldset_manual_v1.csv"
OUT_DEV = "./goldset/goldset_manual_v1_dev.csv"
OUT_TEST = "./goldset/goldset_manual_v1_test.csv"
TEST_RATIO = 0.30


def stable_bucket(key: str) -> float:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    with open(IN_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []

    dev, test = [], []
    for r in rows:
        key = f"{r['pdf_filename']}|{r['sample_id']}|{r['stratum']}"
        if stable_bucket(key) < TEST_RATIO:
            test.append(r)
        else:
            dev.append(r)

    write_csv(OUT_DEV, dev, fieldnames)
    write_csv(OUT_TEST, test, fieldnames)
    print(f"Dev -> {OUT_DEV} ({len(dev)} rows)")
    print(f"Test -> {OUT_TEST} ({len(test)} rows)")


if __name__ == "__main__":
    main()

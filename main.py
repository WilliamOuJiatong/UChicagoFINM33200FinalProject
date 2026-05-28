from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    print("Q&A Credibility Benchmark scaffold is ready.")
    print("Next steps:")
    print("1) Put raw transcript Q&A rows in data/raw/qa_raw.csv")
    print("2) python scripts/make_label_sheet.py --input data/raw/qa_raw.csv --output data/processed/qa_label_sheet.csv")
    print("3) Label and save as data/processed/qa_gold.csv")
    print("4) python scripts/run_benchmark.py --input data/processed/qa_gold.csv --output outputs")
    print(f"Project root: {project_root}")


if __name__ == "__main__":
    main()

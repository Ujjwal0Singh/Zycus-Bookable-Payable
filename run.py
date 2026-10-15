"""run.py — Single-command entrypoint for The Bookable Payable."""
from __future__ import annotations

import glob
import json
import os
import sys

from dotenv import load_dotenv
from erp import erp_book, num
from src.extractor import DocumentSolver
from src.resolver import MasterDataResolver


def main() -> None:
    load_dotenv(override=True)
    force_rerun = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    docs_dir = args[0] if args else "documents"
    out_dir = "output"
    master_dir = "master_data"

    os.makedirs(out_dir, exist_ok=True)
    pdf_files = sorted(glob.glob(os.path.join(docs_dir, "*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in {docs_dir}/", file=sys.stderr)
        sys.exit(1)

    resolver = MasterDataResolver(master_dir=master_dir)
    solver = DocumentSolver(resolver=resolver)

    print(f"Running Bookable Payable Solver over {len(pdf_files)} documents in '{docs_dir}/'...")

    for idx, pdf_path in enumerate(pdf_files, start=1):
        filename = os.path.basename(pdf_path)
        file_stem = os.path.splitext(filename)[0]
        out_path = os.path.join(out_dir, f"{file_stem}.json")

        # Skipping documents that were already processed in previous runs (unless --force is passed)
        if not force_rerun and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            try:
                with open(out_path, encoding="utf-8") as fh:
                    existing = json.load(fh)
                if isinstance(existing, dict) and "payables" in existing:
                    print(f"[{idx:02d}/{len(pdf_files):02d}] Skipping {filename} (already in {out_dir}/)")
                    continue
            except Exception:
                pass  # If the existing JSON is corrupted/partial, re-processing it

        print(f"[{idx:02d}/{len(pdf_files):02d}] Solving {filename} ...", end=" ", flush=True)
        try:
            result = solver.solve_pdf(pdf_path)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)

            n_pay = len(result["payables"])
            if n_pay == 0:
                reasons = ", ".join(d["doc_type"] for d in result["declined"]) or "DECLINED"
                print(f"-> declined ({reasons})")
            else:
                summaries = []
                for p in result["payables"]:
                    booked = erp_book(p)["will_book_gross"]
                    declared = num(p.get("gross_total"))
                    summaries.append(f"booked={booked:.2f} vs printed={declared:.2f} {p.get('currency', '')}")
                print(f"-> {n_pay} payable(s): {'; '.join(summaries)}")
        except Exception as exc:
            print(f"-> ERROR: {exc}", file=sys.stderr)

    print(f"\nCompleted. All outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
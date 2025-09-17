#!/usr/bin/env python3
"""
Bank Statement Parser Agent
---------------------------
Parses bank statement PDFs/CSVs into pandas DataFrames and validates
against a reference CSV. Automatically generates custom parsers.

Usage:
    python agent.py --target icici
"""

import sys
import argparse
import traceback
import importlib
from pathlib import Path

import pandas as pd

# --- Groq API (Optional) ---
try:
    from groq import Groq
    GROQ_API_KEY = "your_groq_api_key_here"
    client = Groq(api_key=GROQ_API_KEY)
except Exception:
    client = None
    print("⚠ Groq client not available. Falling back to default parser template.")

# --- Directories ---
PROJECT_ROOT = Path(__file__).parent.resolve()
PARSERS_DIR = PROJECT_ROOT / "custom_parsers"
DATA_DIR = PROJECT_ROOT / "data"
PARSERS_DIR.mkdir(exist_ok=True)

# --- Expected schema ---
EXPECTED_COLUMNS = ["Date", "Description", "Debit Amt", "Credit Amt", "Balance"]

# --- Default parser template ---
DEFAULT_TEMPLATE = """\
import pandas as pd
import pdfplumber
from pathlib import Path

def parse(file_path: str) -> pd.DataFrame:
    df = pd.DataFrame()
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)

    elif file_path.suffix.lower() == ".pdf":
        rows = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    header = table[0]
                    for row in table[1:]:
                        if row == header or all((cell is None or str(cell).strip() == "") for cell in row):
                            continue
                        rows.append(row)
        if rows:
            df = pd.DataFrame(rows, columns=header)

    # --- Normalize column names ---
    column_map = {
        "Debit": "Debit Amt",
        "Debit Amount": "Debit Amt",
        "Credit": "Credit Amt",
        "Credit Amount": "Credit Amt",
        "Txn Date": "Date",
        "Transaction Date": "Date",
        "Narration": "Description",
        "Details": "Description",
        "Closing Balance": "Balance",
        "Balance Amt": "Balance"
    }
    df = df.rename(columns={c: column_map.get(c.strip(), c.strip()) for c in df.columns})

    # --- Clean numeric columns ---
    for col in ["Debit Amt", "Credit Amt", "Balance"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Clean text columns ---
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("").astype(str).str.strip()

    return df
"""


class BankParserAgent:
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def write_parser(self, bank: str, use_llm=False) -> Path:
        """Write a parser file for the target bank."""
        parser_path = PARSERS_DIR / f"{bank}_parser.py"
        parser_code = DEFAULT_TEMPLATE

        if use_llm and client:
            try:
                prompt = f"""
                Generate a Python parser function parse(file_path:str)->pd.DataFrame
                for {bank} bank statements (CSV/PDF).
                Handle repeated headers, empty rows, numeric cleaning, and ensure
                final DataFrame columns match {EXPECTED_COLUMNS}.
                """
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                parser_code_llm = response.choices[0].message.content
                if parser_code_llm.startswith("```"):
                    parser_code_llm = parser_code_llm.split("```")[-2].strip()
                if "def parse(" in parser_code_llm:
                    parser_code = parser_code_llm
            except Exception as e:
                print(f"⚠ LLM failed, falling back to default template: {e}")

        with open(parser_path, "w", encoding="utf-8") as f:
            f.write(parser_code)
        print(f"✅ Written parser -> {parser_path}")
        return parser_path

    def test_parser(self, bank: str):
        """Run the parser and validate output against reference CSV."""
        sys.path.insert(0, str(PROJECT_ROOT))
        parser_module = importlib.import_module(f"custom_parsers.{bank}_parser")
        parse_func = getattr(parser_module, "parse", None)

        if not parse_func:
            raise AttributeError("parse function not found in parser")

        # Pick file to test
        csv_file = DATA_DIR / bank / "result.csv"
        pdf_file = DATA_DIR / bank / "sample.pdf"
        file_to_parse = csv_file if csv_file.exists() else pdf_file
        if not file_to_parse.exists():
            raise FileNotFoundError(f"No data file found for {bank}")

        # Parse file
        df_out = parse_func(str(file_to_parse))
        df_out.columns = [c.strip() for c in df_out.columns]

        print(f"\n📄 Parsed DataFrame ({len(df_out)} rows, {len(df_out.columns)} cols):")
        print(df_out.head(5))

        # Compare with reference CSV if available
        if csv_file.exists():
            df_ref = pd.read_csv(csv_file)
            df_ref.columns = [c.strip() for c in df_ref.columns]

            # Normalize numeric cols
            for col in ["Debit Amt", "Credit Amt", "Balance"]:
                if col in df_out and col in df_ref:
                    df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0).round(2)
                    df_ref[col] = pd.to_numeric(df_ref[col], errors="coerce").fillna(0).round(2)

            # Normalize text cols
            for col in ["Date", "Description"]:
                if col in df_out and col in df_ref:
                    df_out[col] = df_out[col].astype(str).str.strip()
                    df_ref[col] = df_ref[col].astype(str).str.strip()

            # Check columns
            mismatches = []
            for col in EXPECTED_COLUMNS:
                if col not in df_out.columns or col not in df_ref.columns:
                    mismatches.append(col)
                    continue
                try:
                    pd.testing.assert_series_equal(
                        df_out[col].reset_index(drop=True),
                        df_ref[col].reset_index(drop=True),
                        check_dtype=False,
                    )
                    print(f"✅ Column '{col}' matches")
                except AssertionError:
                    mismatches.append(col)
                    print(f"⚠ Column '{col}' does NOT match")

            if mismatches:
                raise AssertionError(f"Mismatched columns: {mismatches}")

        print("🎉 Parser test passed!\n")
        return True

    def run(self, bank: str):
        """Try generating & testing parser up to N attempts."""
        for attempt in range(1, self.max_attempts + 1):
            print(f"\n🔧 Attempt {attempt}/{self.max_attempts} for '{bank}'...")
            use_llm = attempt > 1
            self.write_parser(bank, use_llm)
            try:
                self.test_parser(bank)
                print(f"✅ Success on attempt {attempt}")
                return
            except Exception as e:
                print(f"❌ Attempt {attempt} failed: {e}")
                traceback.print_exc()
        print(f"🚨 All attempts failed for '{bank}'. Please check your data and parser.")


def main():
    parser = argparse.ArgumentParser(description="Bank Statement Parser Agent")
    parser.add_argument("--target", required=True, help="Bank target (e.g., icici)")
    args = parser.parse_args()

    agent = BankParserAgent()
    agent.run(args.target.strip().lower())


if __name__ == "__main__":
    main()

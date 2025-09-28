import sys
import argparse
import traceback
import importlib
from pathlib import Path
import pandas as pd

# --- Directories ---
PROJECT_ROOT = Path(__file__).parent.resolve()
PARSERS_DIR = PROJECT_ROOT / "custom_parsers"
DATA_DIR = PROJECT_ROOT / "data"
PARSERS_DIR.mkdir(exist_ok=True)

# --- Expected schema ---
EXPECTED_COLUMNS = ["Date", "Description", "Debit Amt", "Credit Amt", "Balance"]

# --- Local LLM imports ---
llm_available = False
LLM_MODEL = None
try:
    from gpt4all import GPT4All

    # Try to use a recent, stable GPT4All model
    candidate_models = [
        "ggml-gpt4all-falcon-q4_0",   # newer stable variant
        "ggml-gpt4all-j-v1.3-groovy", # older
        "ggml-gpt4all-j-v1.2-jazzy",  # oldest
    ]
    for model_name in candidate_models:
        try:
            LLM_MODEL = GPT4All(model_name)
            llm_available = True
            print(f"✅ Using GPT4All model: {model_name}")
            break
        except Exception as e:
            print(f"⚠ Could not load {model_name}: {e}")

except ImportError:
    print("⚠ GPT4All not installed. Falling back to default parser template.")

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

    # Normalize column names
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

    # Clean numeric columns
    for col in ["Debit Amt", "Credit Amt", "Balance"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Clean text columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("").astype(str).str.strip()

    return df
"""

class BankParserAgent:
    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts
        self.model = LLM_MODEL

    def write_parser(self, bank: str, use_llm=False) -> Path:
        """Write a parser file for the target bank."""
        parser_path = PARSERS_DIR / f"{bank}_parser.py"
        parser_code = DEFAULT_TEMPLATE

        if use_llm and llm_available and self.model:
            try:
                prompt = f"""
                Write a Python function `parse(file_path:str)->pd.DataFrame`
                to parse {bank.upper()} bank statements (PDF or CSV).
                Ensure:
                - Handle repeated headers and empty rows
                - Normalize numeric and text columns
                - Final DataFrame columns: {EXPECTED_COLUMNS}
                """
                response = self.model.generate(prompt, n_predict=512)
                code_candidate = response.strip()
                if "def parse(" in code_candidate:
                    parser_code = code_candidate
            except Exception as e:
                print(f"⚠ LLM failed, using default parser: {e}")

        parser_path.write_text(parser_code, encoding="utf-8")
        print(f"✅ Parser written -> {parser_path}")
        return parser_path

    def test_parser(self, bank: str):
        """Run parser and validate against reference CSV."""
        sys.path.insert(0, str(PROJECT_ROOT))
        parser_module = importlib.import_module(f"custom_parsers.{bank}_parser")
        parse_func = getattr(parser_module, "parse", None)
        if not parse_func:
            raise AttributeError("parse function not found in parser")

        # Test files
        csv_file = DATA_DIR / bank / "result.csv"
        pdf_file = DATA_DIR / bank / f"{bank}_sample.pdf"
        file_to_parse = csv_file if csv_file.exists() else pdf_file
        if not file_to_parse.exists():
            raise FileNotFoundError(f"No data file found for {bank}")

        df_out = parse_func(str(file_to_parse))
        df_out.columns = [c.strip() for c in df_out.columns]

        print(f"\n📄 Parsed DataFrame ({len(df_out)} rows, {len(df_out.columns)} cols):")
        print(df_out.head(5))

        # Compare with reference CSV if available
        if csv_file.exists():
            df_ref = pd.read_csv(csv_file)
            df_ref.columns = [c.strip() for c in df_ref.columns]

            for col in ["Debit Amt", "Credit Amt", "Balance"]:
                if col in df_out and col in df_ref:
                    df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0).round(2)
                    df_ref[col] = pd.to_numeric(df_ref[col], errors="coerce").fillna(0).round(2)

            for col in ["Date", "Description"]:
                if col in df_out and col in df_ref:
                    df_out[col] = df_out[col].astype(str).str.strip()
                    df_ref[col] = df_ref[col].astype(str).str.strip()

            mismatches = []
            for col in EXPECTED_COLUMNS:
                if col not in df_out or col not in df_ref:
                    mismatches.append(col)
                    continue
                try:
                    pd.testing.assert_series_equal(
                        df_out[col].reset_index(drop=True),
                        df_ref[col].reset_index(drop=True),
                        check_dtype=False
                    )
                    print(f"✅ Column '{col}' matches")
                except AssertionError:
                    mismatches.append(col)
                    print(f"⚠ Column '{col}' mismatch")

            if mismatches:
                raise AssertionError(f"Mismatched columns: {mismatches}")

        print("🎉 Parser test passed!\n")
        return True

    def run(self, bank: str):
        """Try generating & testing parser up to max_attempts."""
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
        print(f"🚨 All attempts failed for '{bank}'.")

def main():
    parser = argparse.ArgumentParser(description="Bank Statement Parser Agent")
    parser.add_argument("--target", required=True, help="Bank target (e.g., icici)")
    args = parser.parse_args()

    agent = BankParserAgent()
    agent.run(args.target.strip().lower())

if __name__ == "__main__":
    main()
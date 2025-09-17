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

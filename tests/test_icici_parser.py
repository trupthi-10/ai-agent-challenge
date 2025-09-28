import pytest
from custom_parsers.icici_parser import parse
import os

DATA_DIR = os.path.join("data", "icici")
PDF_FILE = os.path.join(DATA_DIR, "icici sample.pdf")
CSV_FILE = os.path.join(DATA_DIR, "result.csv")

def test_parser_creates_csv():
    df = parse(PDF_FILE)
    assert df is not None, "Parser returned None"
    assert not df.empty, "Parser returned empty DataFrame"
    assert list(df.columns) == ["Date", "Description", "Amount", "Balance"], "Columns mismatch"
    assert os.path.exists(CSV_FILE), "CSV file was not created"

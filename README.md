Bank Statement Parser Agent

A Python & Streamlit application to parse bank statement PDFs/CSVs into structured DataFrames, and validate them against reference CSVs. The app can automatically generate custom parsers for different banks.

** Features:

- Upload bank statement PDFs and automatically parse them.
- Supports multiple banks: ICICI, SBI, HDFC, Axis, Kotak (easily extendable).
- Automatically handles repeated headers, empty rows, numeric cleaning, and column normalization.
- Compares parsed data with reference CSVs for validation.
- User-friendly Streamlit interface with preview and debug info.

** Installation:

1. Clone this repository:
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

3. Install dependencies:
pip install -r requirements.txt

** Usage:

Run Streamlit app:
streamlit run app.py

1. Open the URL shown in the terminal (usually http://localhost:8501).
2. Select the bank from the dropdown.
3. Upload a PDF bank statement.
4. Click Run Parser.
5. Parsed data and comparison results (if CSV exists) will be displayed.

Run parser agent from CLI:
python agent.py --target icici

- Automatically generates a parser for the bank.
- Validates parsed data against data/<bank>/result.csv if available.

**  Directory Structure:

project-root/
│
├─ custom_parsers/         # Generated parsers for each bank
├─ data/                   # Bank PDFs and reference CSVs
│   └─ icici/
│       ├─ sample.pdf
│       └─ result.csv
├─ app.py                  # Streamlit interface
├─ agent.py                # Parser agent logic
├─ requirements.txt
└─ README.md

Add New Bank:

1. Upload a sample PDF to data/<bank>/sample.pdf.
2. Add a reference CSV as data/<bank>/result.csv.
3. The agent will generate a custom parser automatically.

Dependencies:

- Python >= 3.9
- pandas
- streamlit
- pdfplumber

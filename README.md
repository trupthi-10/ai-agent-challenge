# 🏦 Bank Statement Parser Agent

**Bank Statement Parser Agent** is a Python project that automatically generates parsers for different banks’ statements (PDF/CSV).
It leverages **GPT4All** (local LLM) or a **default parser template** to normalize raw statements into a clean DataFrame with the following schema:

`Date | Description | Debit Amt | Credit Amt | Balance`

---

## ✨ Features

* 🔄 Supports multiple banks (ICICI, HDFC, SBI, etc.)
* 📑 Handles **PDF** and **CSV** formats
* 🧹 Cleans repeated headers, empty rows, and malformed values
* 🔍 Auto-normalizes columns (Debit, Credit, Balance, Narration → unified format)
* ✅ Includes self-testing with reference CSVs
* 💡 Can extend to new banks by just adding their sample data

---

## 📂 Folder Structure

```plaintext
New challenge/
├── agent.py                  # Main agent script
├── custom_parsers/           # Auto-generated bank parsers (per bank)
│   └── icici_parser.py
├── data/
│   └── icici/
│       ├── icici_sample.pdf  # Sample statement (PDF)
│       └── result.csv        # Expected clean output for testing
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/bank-parser-agent.git
   cd bank-parser-agent
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows)
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) If using GPT4All, make sure to **set your API key / local model path** inside `agent.py`.
   If no LLM is available, the project falls back to a default parser template.

---

## 🚀 Usage

Run the agent for a specific bank (e.g., ICICI):

```bash
python agent.py --target icici
```

This will:

* Generate (or update) a parser in `custom_parsers/icici_parser.py`
* Parse the provided PDF/CSV in `data/icici/`
* Compare results against `data/icici/result.csv`
* Report success or mismatches

---

## 📌 Extending to New Banks

To add support for another bank (e.g., HDFC):

1. Create a folder under `data/hdfc/`
2. Add a sample PDF (`hdfc_sample.pdf`) and reference CSV (`result.csv`)
3. Run:

   ```bash
   python agent.py --target hdfc
   ```
4. A new parser `custom_parsers/hdfc_parser.py` will be generated and tested.

---

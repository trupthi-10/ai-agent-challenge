import streamlit as st
import pandas as pd
import importlib
import traceback
import re
from pathlib import Path
from agent import BankParserAgent, DATA_DIR


BANKS = ["icici", "sbi", "hdfc", "axis", "kotak"]
EXPECTED_COLUMNS = ["Date", "Description", "Debit Amt", "Credit Amt", "Balance"]


st.set_page_config(page_title="Bank Parser Agent", layout="wide")

st.markdown("""
<style>
.stButton>button {
    background-color: #0d6efd;
    color: white;
    font-weight: bold;
    font-size: 16px;
    border-radius: 8px;
    height: 45px;
    width: 180px;
}
.stSelectbox>div>div>div>label,
.stFileUploader>div>label {
    font-weight: bold;
    font-size: 16px;
}
.stDataFrame {
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}
.stAlert {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# --- Page Header ---
st.title("🏦 Bank Statement Parser Agent")
st.markdown(
    "Upload a **bank statement PDF**. The agent will generate a parser "
    "and validate against the reference CSV if available."
)

def canonical_col_name(col: str) -> str:
    if not isinstance(col, str):
        return col
    s = col.strip().lower()
    s = re.sub(r"[\s_]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    if "date" in s or "txn" in s or "transaction" in s:
        return "Date"
    if any(tok in s for tok in ["desc", "narration", "details", "particular"]):
        return "Description"
    if "debit" in s or (("dr" in s.split() and len(s.split()) == 1) or "withdrawal" in s):
        return "Debit Amt"
    if "credit" in s or (("cr" in s.split() and len(s.split()) == 1) or "deposit" in s):
        return "Credit Amt"
    if "balance" in s or "closing" in s:
        return "Balance"
    return col.strip()

def canonicalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [canonical_col_name(c) for c in df.columns]
    return df

def normalize_numeric_cols(df: pd.DataFrame):
    for col in ["Debit Amt", "Credit Amt", "Balance"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round(2)
    return df

def parse_dates(df: pd.DataFrame):
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    if "Date" in df.columns:
        col = df["Date"]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        df["Date_parsed"] = pd.to_datetime(col.astype(str), dayfirst=True, errors="coerce")
    else:
        df["Date_parsed"] = pd.NaT
    return df

def compare_dataframes(df_out: pd.DataFrame, df_ref: pd.DataFrame):
    debug = {}
    df_out = canonicalize_df_columns(df_out)
    df_ref = canonicalize_df_columns(df_ref)

    debug["out_columns"] = df_out.columns.tolist()
    debug["ref_columns"] = df_ref.columns.tolist()

    missing_out = [c for c in EXPECTED_COLUMNS if c not in df_out.columns]
    missing_ref = [c for c in EXPECTED_COLUMNS if c not in df_ref.columns]
    debug["missing_out"] = missing_out
    debug["missing_ref"] = missing_ref
    if missing_out:
        return False, {"reason": "missing_columns_in_parsed", "missing_out": missing_out, "debug": debug}
    if missing_ref:
        return False, {"reason": "missing_columns_in_reference", "missing_ref": missing_ref, "debug": debug}

    df_out = normalize_numeric_cols(df_out)
    df_ref = normalize_numeric_cols(df_ref)
    df_out = parse_dates(df_out)
    df_ref = parse_dates(df_ref)

    debug["out_rows"] = len(df_out)
    debug["ref_rows"] = len(df_ref)

    sort_keys = ["Date_parsed", "Description", "Debit Amt", "Credit Amt", "Balance"]
    df_out_sorted = df_out.sort_values(by=sort_keys, na_position="last").reset_index(drop=True)
    df_ref_sorted = df_ref.sort_values(by=sort_keys, na_position="last").reset_index(drop=True)

    if len(df_out_sorted) == len(df_ref_sorted) and len(df_out_sorted) > 0:
        mismatches = []
        percent_matches = {}
        for col in EXPECTED_COLUMNS:
            s_out = df_out_sorted[col].fillna("").astype(str)
            s_ref = df_ref_sorted[col].fillna("").astype(str)
            matches = (s_out == s_ref)
            pct = matches.mean()
            percent_matches[col] = float(pct)
            if pct < 0.98:
                mismatches.append({"column": col, "match_pct": pct})
        debug["percent_matches"] = percent_matches
        if not mismatches:
            return True, {"reason": "exact_row_match", "debug": debug}
        else:
            return False, {"reason": "row_values_mismatch", "mismatches": mismatches, "debug": debug}

    key_cols = ["Date_parsed", "Debit Amt", "Credit Amt", "Balance"]
    df_out_keyed = df_out.dropna(subset=["Date_parsed"])[key_cols].copy()
    df_ref_keyed = df_ref.dropna(subset=["Date_parsed"])[key_cols].copy()
    merged = pd.merge(
        df_out_keyed.assign(_out_idx=range(len(df_out_keyed))),
        df_ref_keyed.assign(_ref_idx=range(len(df_ref_keyed))),
        on=key_cols,
        how="inner",
    )
    matched = len(merged)
    debug["matched_rows_by_key"] = matched
    min_rows = min(len(df_out_keyed), len(df_ref_keyed)) if (len(df_out_keyed) and len(df_ref_keyed)) else 0
    match_rate = matched / min_rows if min_rows > 0 else 0.0
    debug["match_rate_rows"] = float(match_rate)
    if match_rate >= 0.8 and matched > 0:
        return True, {"reason": "row_key_match", "matched": matched, "match_rate": float(match_rate), "debug": debug}

    col_mismatch_info = []
    for col in EXPECTED_COLUMNS:
        s_out = df_out[col].fillna("").astype(str)
        s_ref = df_ref[col].fillna("").astype(str)
        common = set(s_out.tolist()) & set(s_ref.tolist())
        prop_common = len(common) / max(1, min(len(s_out.unique()), len(s_ref.unique())))
        col_mismatch_info.append({"column": col, "prop_common_unique": float(prop_common)})
    debug["col_loose_info"] = col_mismatch_info

    return False, {"reason": "no_sufficient_match", "col_loose_info": col_mismatch_info, "debug": debug}

# --- UI Layout (Card Style) ---
with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Upload & Select Bank")
        bank_name = st.selectbox("Select Bank", BANKS)
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
        run_parser = st.button("Run Parser")
    with col2:
        st.subheader("Instructions")
        st.markdown("""
        1. Select the bank from the dropdown.
        2. Upload the bank statement PDF.
        3. Click **Run Parser** to parse and validate.
        4. Parsed DataFrame and comparison (if CSV exists) will be shown below.
        """)

if run_parser:
    if not bank_name:
        st.error("⚠ Please select a bank.")
    elif not uploaded_pdf:
        st.error("⚠ Please upload a PDF file.")
    else:
        try:
            # Save PDF
            bank_dir = DATA_DIR / bank_name
            bank_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = bank_dir / "sample.pdf"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            # Generate parser
            agent = BankParserAgent()
            agent.write_parser(bank_name, use_llm=False)

            parser_module = importlib.import_module(f"custom_parsers.{bank_name}_parser")
            parse_func = getattr(parser_module, "parse", None)
            if not parse_func:
                st.error("❌ Generated parser missing `parse` function.")
                st.stop()

            df_out = parse_func(str(pdf_path))
            if df_out is None or not isinstance(df_out, pd.DataFrame):
                st.error("❌ Parser did not return a DataFrame.")
                st.stop()

            df_out = canonicalize_df_columns(df_out)
            df_out = normalize_numeric_cols(df_out)
            df_out = parse_dates(df_out)

            st.subheader("📄 Parsed DataFrame Preview")
            st.dataframe(df_out.head(20), use_container_width=True)

            # Compare with reference CSV if exists
            csv_file = bank_dir / "result.csv"
            if csv_file.exists():
                df_ref = pd.read_csv(csv_file)
                df_ref = canonicalize_df_columns(df_ref)
                df_ref = normalize_numeric_cols(df_ref)
                df_ref = parse_dates(df_ref)

                success, details = compare_dataframes(df_out, df_ref)
                if success:
                    st.success("✅ Parsed output matches reference.")
                    with st.expander("Debug Info"):
                        st.json(details.get("debug", {}))
                else:
                    st.warning("⚠ Parsed output did NOT match reference.")
                    with st.expander("Debug Info"):
                        st.json(details.get("debug", {}))
                    st.write("Parsed columns:", df_out.columns.tolist())
                    st.write("Reference columns:", df_ref.columns.tolist())
            else:
                st.info("ℹ No reference CSV found. Only parsed output is shown.")

        except Exception as e:
            st.error(f"❌ Error running parser: {e}")
            st.code(traceback.format_exc())


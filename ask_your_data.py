"""
ask_your_data.py — Natural-language querying tab for RetailPulse.

Lets a user upload their own CSV/Excel file, asks a plain-English question,
converts it to a DuckDB SQL query via an LLM (OpenAI or Gemini), executes it,
and displays results with an auto-generated chart where appropriate.

Design notes:
- The LLM call is isolated behind `generate_sql()` so it can be swapped/mocked
  for testing without touching the surrounding upload/execute/display logic.
- DuckDB queries the in-memory pandas dataframe directly (`FROM df`) — no
  actual database server or file needed.
- The generated SQL is always shown to the user (SQL explanation mode) so
  nothing runs as an unreviewed black box.
"""
import streamlit as st
import pandas as pd
import duckdb
import io
import re
from datetime import datetime


# ------------------------------------------------------------------
# File loading with encoding fallback
# ------------------------------------------------------------------
def load_uploaded_file(uploaded_file):
    """Reads a CSV or Excel upload, trying multiple encodings for CSV since
    real-world exports are frequently not UTF-8 (Windows exports are often
    cp1252/latin1)."""
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_error = None
    for enc in encodings_to_try:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
    raise ValueError(f"Could not decode file with any of {encodings_to_try}: {last_error}")


# ------------------------------------------------------------------
# Schema detection
# ------------------------------------------------------------------
def detect_schema(df: pd.DataFrame) -> pd.DataFrame:
    schema = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(t) for t in df.dtypes],
        "Sample": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "" for c in df.columns],
        "Nulls": df.isna().sum().values,
    })
    return schema


def _column_kind(df, col):
    if pd.api.types.is_numeric_dtype(df[col]):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return "date"
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.to_datetime(df[col].dropna().head(20))
        return "date"
    except Exception:
        return "categorical"


# ------------------------------------------------------------------
# Example question suggestions (heuristic, no LLM call needed)
# ------------------------------------------------------------------
def suggest_example_questions(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if _column_kind(df, c) == "numeric"]
    cat_cols = [c for c in df.columns if _column_kind(df, c) == "categorical"]
    date_cols = [c for c in df.columns if _column_kind(df, c) == "date"]

    suggestions = []
    if numeric_cols:
        suggestions.append(f"What is the total {numeric_cols[0]}?")
    if numeric_cols and cat_cols:
        suggestions.append(f"Show total {numeric_cols[0]} by {cat_cols[0]}")
        suggestions.append(f"Which {cat_cols[0]} has the highest {numeric_cols[0]}?")
    if numeric_cols and date_cols:
        suggestions.append(f"Show {numeric_cols[0]} over time by {date_cols[0]}")
    if cat_cols:
        suggestions.append(f"How many rows are there per {cat_cols[0]}?")
    suggestions.append("Show the top 10 rows sorted by the largest numeric column")
    return suggestions[:5]


# ------------------------------------------------------------------
# SQL generation (LLM-backed, isolated for testability)
# ------------------------------------------------------------------
def _build_prompt(question: str, schema: pd.DataFrame) -> str:
    schema_text = "\n".join(f"- {r.Column} ({r.Type})" for r in schema.itertuples())
    return f"""You are a SQL assistant. The user has a table named `df` in DuckDB with this schema:
{schema_text}

Write a single DuckDB SQL query (no explanation, no markdown fences, just the SQL)
that answers this question: "{question}"

Rules:
- Query only the table `df`.
- Return a SELECT statement only (no DDL/DML).
- Use standard SQL DuckDB supports.
"""


def generate_sql(question: str, schema: pd.DataFrame, provider: str, api_key: str,
                  _llm_call=None) -> str:
    """Generates a SQL query for the given question.

    `_llm_call` is an injectable function (signature: prompt -> str) used for
    testing without hitting a real API. In production it is left as None and
    the real OpenAI/Gemini client is used based on `provider`.
    """
    prompt = _build_prompt(question, schema)

    if _llm_call is not None:
        raw = _llm_call(prompt)
    elif provider == "OpenAI":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content
    elif provider == "Google Gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        raw = model.generate_content(prompt).text
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return _clean_sql(raw)


def _clean_sql(raw: str) -> str:
    """Strips markdown fences and leading/trailing noise an LLM might add,
    since even with 'no markdown fences' in the prompt, models sometimes
    wrap the answer in ```sql ... ``` anyway."""
    text = raw.strip()
    text = re.sub(r"^```(sql)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def is_safe_select(sql: str) -> bool:
    """Guards against anything other than a read-only SELECT -- the LLM
    should never be allowed to run DDL/DML against the uploaded data."""
    normalized = sql.strip().lower()
    if not normalized.startswith("select") and not normalized.startswith("with"):
        return False
    forbidden = ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "attach ", "copy "]
    return not any(f in normalized for f in forbidden)


# ------------------------------------------------------------------
# Query execution
# ------------------------------------------------------------------
def run_query(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("df", df)
    result = con.execute(sql).fetchdf()
    con.close()
    return result


# ------------------------------------------------------------------
# Auto-charting
# ------------------------------------------------------------------
def auto_chart(result: pd.DataFrame):
    """Renders a bar/line chart automatically when the result shape makes
    it obvious which chart fits -- one label column plus one numeric
    column. Falls back to no chart (table only) otherwise."""
    if result.empty or len(result.columns) < 2:
        return
    numeric_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
    label_cols = [c for c in result.columns if c not in numeric_cols]
    if not numeric_cols or not label_cols:
        return

    label_col, value_col = label_cols[0], numeric_cols[0]
    chart_df = result.set_index(label_col)[[value_col]]

    is_date_like = False
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.to_datetime(result[label_col].head(5))
        is_date_like = True
    except Exception:
        pass

    if is_date_like:
        st.line_chart(chart_df)
    elif len(result) <= 25:
        st.bar_chart(chart_df)


# ------------------------------------------------------------------
# Export helpers
# ------------------------------------------------------------------
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return buf.getvalue()


# ------------------------------------------------------------------
# Main tab render function
# ------------------------------------------------------------------
def render_ask_your_data_tab():
    st.subheader("💬 Ask Your Data")
    st.caption("Upload your own dataset, ask a question in plain English, get an instant answer — no SQL required.")

    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    with st.sidebar:
        st.markdown("### Ask Your Data — Settings")
        provider = st.selectbox("AI Provider", ["OpenAI", "Google Gemini"])
        api_key = st.text_input(f"{provider} API Key", type="password",
                                 help="Your key is used only for this session and is never saved.")

    uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"], key="ayd_upload")
    if not uploaded:
        st.info("Upload a dataset above to get started.")
        return

    try:
        df = load_uploaded_file(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")
    with st.expander("📋 Schema & Preview", expanded=False):
        st.dataframe(detect_schema(df), hide_index=True, use_container_width=True)
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("**Try an example question:**")
    examples = suggest_example_questions(df)
    cols = st.columns(len(examples))
    example_clicked = None
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            example_clicked = ex

    question = st.text_input("Or ask your own question:", value=example_clicked or "")

    explain_mode = st.checkbox("🔍 SQL explanation mode (show what the query does, in plain English)")

    if st.button("Ask", type="primary") and question:
        if not api_key:
            st.warning(f"Enter your {provider} API key in the sidebar first.")
            return
        schema = detect_schema(df)
        with st.spinner("Generating SQL..."):
            try:
                sql = generate_sql(question, schema, provider, api_key)
            except Exception as e:
                st.error(f"Could not generate SQL: {e}")
                return

        if not is_safe_select(sql):
            st.error("The generated query wasn't a safe read-only SELECT, so it was blocked. Try rephrasing your question.")
            st.code(sql, language="sql")
            return

        st.markdown("**Generated SQL:**")
        st.code(sql, language="sql")

        if explain_mode:
            st.info(_plain_english_explanation(sql))

        try:
            result = run_query(df, sql)
        except Exception as e:
            st.error(f"Query failed to run: {e}")
            return

        st.markdown("**Result:**")
        st.dataframe(result, use_container_width=True)
        auto_chart(result)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download CSV", result.to_csv(index=False).encode("utf-8"),
                                "query_result.csv", "text/csv", use_container_width=True)
        with col2:
            st.download_button("⬇️ Download Excel", to_excel_bytes(result),
                                "query_result.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True)

        st.session_state.query_history.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "question": question,
            "sql": sql,
            "rows": len(result),
        })

    if st.session_state.query_history:
        with st.expander(f"💾 Query History ({len(st.session_state.query_history)})"):
            for item in st.session_state.query_history:
                st.markdown(f"**[{item['time']}]** {item['question']} — {item['rows']} rows")
                st.code(item["sql"], language="sql")


def _plain_english_explanation(sql: str) -> str:
    """A simple, rule-based explanation as a fallback/complement to an
    LLM-generated one -- keeps explanation mode useful even without an
    extra API call for every query."""
    s = sql.lower()
    parts = []
    if "group by" in s:
        parts.append("groups rows together")
    if "order by" in s:
        parts.append("sorts the results")
    if "limit" in s:
        parts.append("limits how many rows come back")
    if "sum(" in s:
        parts.append("adds up a numeric column")
    if "count(" in s:
        parts.append("counts rows")
    if "avg(" in s:
        parts.append("averages a numeric column")
    if not parts:
        return "This query selects and filters rows directly from your data without aggregation."
    return "This query " + ", ".join(parts) + "."

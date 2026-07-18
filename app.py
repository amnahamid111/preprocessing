import streamlit as st
import pandas as pd

# Import your real functions
# from utils import analyze_dataset, generate_recommendations, apply_recommendations

st.set_page_config(
    page_title="ML Preprocessing Assistant",
    layout="wide"
)

st.title("🤖 ML Preprocessing Assistant")

uploaded_file = st.file_uploader(
    "📂 Upload Dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # Read dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # -------------------------
    # Original Dataset
    # -------------------------

    st.subheader("📄 Original Dataset")

    st.dataframe(df)

    # -------------------------
    # Analyze Dataset
    # -------------------------

    report = analyze_dataset(df)

    st.subheader("📊 Analysis Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", report["rows"])
    col2.metric("Columns", report["columns"])
    col3.metric("Duplicates", report["duplicate_rows"])
    col4.metric(
        "Missing Values",
        sum(report["missing_values"].values())
    )

    st.subheader("📋 Dataset Report")

    st.write("### Data Types")
    st.json(report["data_types"])

    st.write("### Missing Values")
    st.json(report["missing_values"])

    st.write("### Numerical Columns")
    st.write(report["numerical_columns"])

    st.write("### Categorical Columns")
    st.write(report["categorical_columns"])

    # -------------------------
    # Recommendations
    # -------------------------

    recommendations = generate_recommendations(report)

    st.subheader("💡 Recommendations")

    for rec in recommendations:

        if isinstance(rec, str):

            st.info(rec)

        else:

            st.success(
                f"""
**Column:** {rec['column']}

**Issue:** {rec['issue']}

**Severity:** {rec['severity']}

**Recommendation:** {rec['recommendation']}
"""
            )

    # -------------------------
    # Apply Recommendations
    # -------------------------

    processed_df = apply_recommendations(
        df,
        recommendations
    )

    st.success("✅ Recommendations applied successfully!")

    # -------------------------
    # Processed Dataset
    # -------------------------

    st.subheader("🧹 Processed Dataset")

    st.write(
        f"""
Rows : **{processed_df.shape[0]}**

Columns : **{processed_df.shape[1]}**
"""
    )

    st.dataframe(processed_df)

    # -------------------------
    # Download
    # -------------------------

    csv = processed_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Cleaned Dataset",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

else:

    st.info("📂 Please upload a CSV or Excel dataset.")

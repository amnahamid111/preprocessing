import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# -----------------------------
# Analyze Dataset
# -----------------------------
def analyze_dataset(df):

    report = {}

    # Basic Information
    report["rows"] = df.shape[0]
    report["columns"] = df.shape[1]
    report["column_names"] = df.columns.tolist()

    report["data_types"] = {
        col: str(dtype)
        for col, dtype in df.dtypes.items()
    }

    # Missing Values
    report["missing_values"] = df.isnull().sum().to_dict()
    report["missing_percentage"] = (
        df.isnull().sum() / len(df)
    ).to_dict()

    # Duplicate Rows
    report["duplicate_rows"] = df.duplicated().sum()

    # Numerical Columns
    report["numerical_columns"] = (
        df.select_dtypes(include="number")
        .columns
        .tolist()
    )

    # Categorical Columns
    report["categorical_columns"] = (
        df.select_dtypes(include="object")
        .columns
        .tolist()
    )

    # Unique Values
    report["unique_values"] = df.nunique().to_dict()

    return 
    
# -----------------------------
# Generate Recommendations
# -----------------------------
def generate_recommendations(report):

    recommendations = []

    # -----------------------------
    # Duplicate Rows
    # -----------------------------
    if report["duplicate_rows"] > 0:

        recommendations.append({
            "issue": "Duplicate Rows",
            "severity": "High",
            "recommendation": "Remove duplicate rows."
        })

    else:

        recommendations.append({
            "issue": "Duplicate Rows",
            "severity": "Low",
            "recommendation": "No duplicate rows found."
        })

    # -----------------------------
    # Missing Values
    # -----------------------------
    for column, percent in report["missing_percentage"].items():

        if percent > 0:

            recommendations.append({

                "column": column,
                "issue": "Missing Values",
                "severity": "Medium",
                "recommendation": "Fill missing values with Mode"

            })

    # -----------------------------
    # ID Columns
    # -----------------------------
    rows = report["rows"]

    for column, unique_count in report["unique_values"].items():

        if unique_count == rows:

            recommendations.append({

                "column": column,
                "issue": "Possible ID Column",
                "severity": "Medium",
                "recommendation": "Exclude from model training"

            })

    # -----------------------------
    # Constant Columns
    # -----------------------------
    for column, unique_count in report["unique_values"].items():

        if unique_count == 1:

            recommendations.append({

                "column": column,
                "issue": "Constant Column",
                "severity": "Medium",
                "recommendation": "Consider removing this column."

            })

    # -----------------------------
    # Encoding Recommendation
    # -----------------------------
    for column in report["categorical_columns"]:

        unique_count = report["unique_values"][column]

        # Skills Column
        if column.lower() == "skills":

            recommendations.append({

                "column": column,
                "issue": "Categorical Column",
                "severity": "High",
                "recommendation": "Use MultiLabelBinarizer"

            })

        elif unique_count == 2:

            recommendations.append({

                "column": column,
                "issue": "Categorical Column",
                "severity": "Medium",
                "recommendation": "Use Label Encoding"

            })

        else:

            recommendations.append({

                "column": column,
                "issue": "Categorical Column",
                "severity": "Medium",
                "recommendation": "Use One-Hot Encoding"

            })

    return recommendations

# -----------------------------
# Apply Recommendations
# -----------------------------
def apply_recommendations(df, recommendations):

    df = df.copy()

    le = LabelEncoder()

    for rec in recommendations:

        # Skip recommendations without a column
        if "column" not in rec:
            continue

        column = rec["column"]
        action = rec["recommendation"]

        # Skip if column no longer exists
        if column not in df.columns:
            continue

        # ------------------------------------
        # Remove ID Columns
        # ------------------------------------
        if action == "Exclude from model training":

            df = df.drop(columns=[column])

        # ------------------------------------
        # Fill Missing Values
        # ------------------------------------
        elif action == "Fill missing values with Mode":

            if df[column].isnull().sum() > 0:

                mode = df[column].mode()

                if len(mode) > 0:
                    df[column] = df[column].fillna(mode[0])

        # ------------------------------------
        # Label Encoding
        # ------------------------------------
        elif action == "Use Label Encoding":

            df[column] = le.fit_transform(df[column].astype(str))

        # ------------------------------------
        # One-Hot Encoding
        # ------------------------------------
        elif action == "Use One-Hot Encoding":

            df = pd.get_dummies(
                df,
                columns=[column],
                drop_first=True
            )

        # ------------------------------------
        # MultiLabelBinarizer Placeholder
        # ------------------------------------
        elif action == "Use MultiLabelBinarizer":

            # For now we'll skip this.
            # You can improve it later.
            pass

    return df

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HomeValue | House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# =========================================================
# DATASET
# =========================================================

DATA = Path(__file__).parent / "data" / "train.csv"


if not DATA.exists():
    st.error(
        "train.csv not found. Please keep train.csv inside the data folder."
    )
    st.stop()


# =========================================================
# TRAIN MACHINE LEARNING MODEL
# =========================================================

@st.cache_resource
def train_model():

    df = pd.read_csv(DATA)

    features = [
        "GrLivArea",
        "OverallQual",
        "GarageCars",
        "TotalBsmtSF",
        "YearBuilt",
        "FullBath",
        "BedroomAbvGr"
    ]

    X = df[features]
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                min_samples_leaf=2
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return pipeline, features, mae, rmse, r2


model, features, mae, rmse, r2 = train_model()


# =========================================================
# HEADER
# =========================================================

st.title("🏠 HomeValue")

st.write(
    "### AI-Powered House Price Prediction"
)

st.write(
    "Estimate residential property value using a "
    "machine-learning model trained on historical housing data."
)

st.info(
    "🤖 Random Forest Regression  •  Real Housing Dataset  •  Streamlit"
)


# =========================================================
# PROPERTY INPUT
# =========================================================

st.divider()

st.header("🏡 Property Profile")

st.write(
    "Enter the property details below to generate an estimated sale price."
)


col1, col2 = st.columns(2)


# ---------------- LEFT COLUMN ----------------

with col1:

    living_area = st.number_input(
        "Living Area (sq ft)",
        min_value=300,
        max_value=10000,
        value=1600,
        step=50
    )

    quality = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=6
    )

    garage = st.selectbox(
        "Garage Capacity",
        [0, 1, 2, 3, 4],
        index=2,
        format_func=lambda x:
        f"{x} car" if x == 1 else f"{x} cars"
    )

    basement = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=800,
        step=50
    )


# ---------------- RIGHT COLUMN ----------------

with col2:

    year_built = st.number_input(
        "Year Built",
        min_value=1870,
        max_value=2026,
        value=2000,
        step=1
    )

    bathrooms = st.selectbox(
        "Full Bathrooms",
        [0, 1, 2, 3, 4],
        index=2
    )

    bedrooms = st.selectbox(
        "Bedrooms",
        [1, 2, 3, 4, 5, 6, 7, 8],
        index=2
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

predict = st.button(
    "🔮 Estimate Property Value",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION RESULT
# =========================================================

st.divider()

st.header("💰 Estimated Valuation")


if predict:

    input_data = pd.DataFrame(
        {
            "GrLivArea": [living_area],
            "OverallQual": [quality],
            "GarageCars": [garage],
            "TotalBsmtSF": [basement],
            "YearBuilt": [year_built],
            "FullBath": [bathrooms],
            "BedroomAbvGr": [bedrooms]
        }
    )

    prediction = float(
        model.predict(input_data)[0]
    )

    st.success(
        "Prediction generated successfully!"
    )

    st.metric(
        label="Estimated Sale Price",
        value=f"${prediction:,.0f}"
    )

    st.write("### Property Summary")

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric(
            "Living Area",
            f"{living_area:,} sq ft"
        )

    with s2:
        st.metric(
            "Bedrooms",
            bedrooms
        )

    with s3:
        st.metric(
            "Bathrooms",
            bathrooms
        )

    with s4:
        st.metric(
            "Quality",
            f"{quality}/10"
        )

else:

    st.info(
        "Enter the property details above and click "
        "'Estimate Property Value' to generate a prediction."
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.header("📊 Model Performance")

st.caption(
    "Performance measured on a held-out 20% validation set."
)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "MAE",
        f"${mae:,.0f}"
    )

with m2:
    st.metric(
        "RMSE",
        f"${rmse:,.0f}"
    )

with m3:
    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.header("🔍 About This Project")

a, b, c = st.columns(3)


with a:

    st.subheader("📊 Dataset")

    st.write(
        "Ames, Iowa historical residential "
        "property sales dataset."
    )


with b:

    st.subheader("🤖 Machine Learning")

    st.write(
        "Random Forest Regression with "
        "missing-value handling and preprocessing."
    )


with c:

    st.subheader("⚙️ Technology")

    st.write(
        "Python • Pandas • NumPy • "
        "Scikit-learn • Streamlit"
    )


# =========================================================
# FEATURES USED
# =========================================================

st.divider()

st.subheader("🧠 Features Used by the Model")

st.write(
    "The model uses the following property characteristics:"
)

st.write(
    "• Living Area  • Overall Quality  • Garage Capacity  "
    "• Basement Area  • Year Built  • Bathrooms  • Bedrooms"
)


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.warning(
    "This project uses historical Ames, Iowa housing data. "
    "The predictions are estimates for educational purposes "
    "and are not formal property appraisals. The model is "
    "not calibrated for Indian real-estate markets."
)


# =========================================================
# FOOTER
# =========================================================
st.caption(
    "🏠 HomeValue | AI Property Intelligence | "
    "made by isha"
)
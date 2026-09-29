# HomeValue — House Price Prediction

An end-to-end machine-learning web app that estimates residential sale prices from property characteristics.

## Problem statement
Home buyers, sellers, and analysts often need a starting estimate of a property's value. This project learns relationships between property attributes and historical sale prices, then predicts an estimated sale price for a supplied property profile.

**Important:** The dataset contains historical sales from Ames, Iowa, USA. The model is educational and should not be used to value Indian properties or make financial decisions.

## Dataset
Kaggle's [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). The training file has 1,460 properties and includes `SalePrice`, the target variable.

Download `train.csv` from Kaggle and place it at `data/train.csv`. The dataset is not included in this repository; Kaggle may require a free account and acceptance of competition rules.

## Features used
- `GrLivArea`: above-ground living area (sq ft)
- `OverallQual`: overall material and finish quality (1–10)
- `GarageCars`: garage capacity
- `TotalBsmtSF`: basement area (sq ft)
- `YearBuilt`: original construction year
- `FullBath`: full bathrooms above ground
- `BedroomAbvGr`: bedrooms above basement level

## Workflow
1. Load and inspect real historical sales data.
2. Select relevant numeric features and split data into training and validation sets.
3. Impute missing values using the training pipeline.
4. Train a Random Forest regression model.
5. Evaluate on a held-out 20% split using MAE, RMSE and R².
6. Serve predictions through a Streamlit interface.

The app trains the model on first load and caches it for subsequent interactions.

## Run locally (Windows)
1. Install Python 3.10 or newer.
2. Download `train.csv` from Kaggle and put it in the `data` folder.
3. Open a terminal in this project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```text
house_price_prediction/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── data/
    └── train.csv  # download separately; not committed
```

## Evaluation
The app displays MAE (average absolute error in dollars), RMSE (penalizes larger errors more) and R² (variance explained) for a held-out validation split. Actual values are generated when you run the project; no performance result is claimed in advance.

## Limitations
- Ames, Iowa data is not representative of other cities or countries.
- The model uses seven features and omits many potentially relevant factors.
- Estimates depend on the dataset's historical price range and market conditions.
- A random split is a basic validation method, not proof of future-market accuracy.

## GitHub
Create a new public repository named `house-price-prediction`, then push this folder. See the Git commands in the project setup guide provided in the chat.

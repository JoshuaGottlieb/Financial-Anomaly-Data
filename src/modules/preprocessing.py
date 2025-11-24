from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def extract_derived_temporal_features(
    df: pd.DataFrame,
    categorical_columns: List[str] = None,
    amount_col: str = "amount",
    timestamp_col: str = "timestamp",
    day_col: str = "day",
    hour_col: str = "hour"
) -> pd.DataFrame:
    """
    Extract derived temporal features for categorical grouping columns, including:
      - Time delta between consecutive transactions per category
      - Amount delta between consecutive transactions per category
      - Hourly count velocity per category
      - Hourly sum velocity per category

    Args:
        df (pd.DataFrame): Input dataframe containing transaction-level data.
        categorical_columns (List[str], optional): Columns to group by (e.g., ['account', 'merchant']).
        amount_col (str): Name of the column with transaction amounts.
        timestamp_col (str): Name of the timestamp column.
        day_col (str): Column representing the day (used for hourly grouping).
        hour_col (str): Column representing the hour (used for hourly grouping).

    Returns:
        pd.DataFrame: The input dataframe augmented with all derived temporal features.
    """
    if categorical_columns is None:
        raise ValueError("categorical_columns must be provided.")

    df = df.copy()

    for col in categorical_columns:

        # --- 1. TIME DELTA + AMOUNT DELTA ---
        col_time_delta = pd.DataFrame()
        col_amount_delta = pd.DataFrame()

        for val in df[col].unique():
            sub = df[df[col] == val].sort_values(timestamp_col)

            # time delta in minutes
            sub_time_delta = (
                sub[timestamp_col]
                .diff(1)
                .fillna(pd.Timedelta(0))
                .dt.total_seconds() / 60
            ).to_frame(name = f"{col}_time_delta")

            # amount delta
            sub_amount_delta = (
                sub[amount_col]
                .diff(1)
                .fillna(0)
                .to_frame(name = f"{col}_amount_delta")
            )

            col_time_delta = pd.concat([col_time_delta, sub_time_delta], axis = 0)
            col_amount_delta = pd.concat([col_amount_delta, sub_amount_delta], axis = 0)

        col_time_delta = col_time_delta.sort_index().astype(int)
        col_amount_delta = col_amount_delta.sort_index()

        # --- 2. HOURLY COUNT VELOCITY ---
        col_count_velocity = (
            df.groupby([col, day_col, hour_col])[amount_col]
            .count()
            .reset_index()
            .rename(columns = {amount_col: f"{col}_hourly_transaction_count"})
        )

        # --- 3. HOURLY SUM VELOCITY ---
        col_amount_velocity = (
            df.groupby([col, day_col, hour_col])[amount_col]
            .sum()
            .reset_index()
            .rename(columns = {amount_col: f"{col}_hourly_transaction_sum"})
        )

        # --- 4. MERGE NEW FEATURES BACK TO DATAFRAME ---
        df = pd.concat([df, col_time_delta, col_amount_delta], axis = 1)
        df = pd.merge(
            df, col_count_velocity, on = [col, day_col, hour_col], how = "inner"
        )
        df = pd.merge(
            df, col_amount_velocity, on = [col, day_col, hour_col], how = "inner"
        )

    return df

def encode_features(
    X: pd.DataFrame,
    y: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    passthrough_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and testing subsets and apply column-specific
    preprocessing transformations. Although this function is used in an 
    unsupervised learning context, the `y` variable is included only for the 
    purposes of stratification during the split and for downstream evaluation.

    The transformation pipeline includes:
        - Scaling numeric columns using `MinMaxScaler`
        - One-hot encoding categorical columns using `OneHotEncoder`
        - Passing through selected columns unchanged

    Args:
        X (pd.DataFrame):
            Dataframe containing all feature columns.
        y (pd.DataFrame):
            Variable used only for stratifying the train/test split and for
            later evaluation. It is not used as a supervised target during fitting.
        numeric_cols (List[str]):
            Columns to be scaled using `MinMaxScaler`.
        categorical_cols (List[str]):
            Columns to be one-hot encoded using `OneHotEncoder`.
        passthrough_cols (List[str]):
            Columns to be included in the output without transformation.
        test_size (float, optional):
            Fraction of the data reserved for the test set. Defaults to 0.2.
        random_state (int, optional):
            Random seed for reproducibility of the split. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            A tuple `(train, test)` where each dataframe contains the transformed
            feature columns along with the appended `y` variable.
    """

    # Train/test split (stratified using y for evaluation purposes)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = test_size,
        random_state = random_state,
        stratify = y
    )

    # Column-specific preprocessing: scaling, encoding, passthrough
    preprocessor = ColumnTransformer(
        transformers = [
            ('minmax', MinMaxScaler(), numeric_cols),               # Scale numeric features
            ('pass', 'passthrough', passthrough_cols),              # Keep passthrough features unchanged
            ('ohe', OneHotEncoder(drop = None), categorical_cols)   # One-hot encode categorical
        ],
        sparse_threshold = 0,             # Force dense numpy arrays
        n_jobs = -1,                      # Parallel processing
        verbose_feature_names_out = False # Cleaner feature names
    )

    # Fit on training set and transform both training and test data
    X_train_tr = preprocessor.fit_transform(X_train, y_train)
    X_train_tr = pd.DataFrame(
        X_train_tr,
        index = X_train.index,
        columns = preprocessor.get_feature_names_out()
    )

    X_test_tr = preprocessor.transform(X_test)
    X_test_tr = pd.DataFrame(
        X_test_tr,
        index = X_test.index,
        columns = preprocessor.get_feature_names_out()
    )

    # Reattach y for evaluation
    train = pd.concat([X_train_tr, y_train], axis = 1)
    test = pd.concat([X_test_tr, y_test], axis = 1)

    return train, test
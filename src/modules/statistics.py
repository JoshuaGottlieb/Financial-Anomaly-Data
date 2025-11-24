from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kruskal

def calculate_VIF(
    dataframe: pd.DataFrame,
    columns: Optional[List[str]] = None,
    log1p_columns: Optional[List[str]] = None,
    ridge: float = 1e-8,
    verbose: bool = False
) -> pd.Series:
    """
    Calculate Variance Inflation Factors (VIF) for numeric columns in a DataFrame.

    VIF is calculated as the diagonal of the inverse of the correlation matrix:
        VIF_j = diag(inv(corr_matrix))

    This version handles singular matrices by adding a small ridge term.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        columns (List[str], optional): Subset of columns to compute VIF for.
            Defaults to all numeric columns.
        log1p_columns (List[str], optional): List of columns to apply np.log1p transformation
            before computing correlation. Defaults to None.
        ridge (float, optional): Small value to add to diagonal to handle singular matrices. Defaults to 1e-8.
        verbose (bool, optional): If True, print warnings for high VIF (>10). Defaults to False.

    Returns:
        pd.Series: VIF values indexed by column names, series name "VIF".
    """
    # Select numeric columns
    numeric_cols = dataframe.select_dtypes(include = np.number).columns.tolist()
    
    if columns is not None:
        numeric_cols = [col for col in columns if col in numeric_cols]

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns available for VIF calculation.")

    # Copy and transform log1p columns if specified
    df_vif = dataframe[numeric_cols].copy()
    if log1p_columns:
        for col in log1p_columns:
            if col in df_vif.columns:
                df_vif[col] = np.log1p(df_vif[col])
    
    # Compute correlation matrix
    corr_matrix = df_vif.corr(numeric_only = True)
    
    # Regularize diagonal to handle singular matrices
    corr_matrix += np.eye(len(corr_matrix)) * ridge

    # Invert correlation matrix
    inv_corr_matrix = np.linalg.inv(corr_matrix.values)
    
    # VIF is the diagonal of the inverse correlation matrix
    vif_values = pd.Series(np.diag(inv_corr_matrix), index = corr_matrix.columns, name = "VIF")
    
    # Verbose warnings for high VIF
    if verbose:
        high_vif = vif_values[vif_values > 10]
        if not high_vif.empty:
            print("Warning: High multicollinearity detected. Variables with VIF > 10:")
            for col, val in high_vif.items():
                print(f"  - {col}: {val:.2f}")
    
    return vif_values

def compute_kruskal_wallis(
    dataframe: pd.DataFrame,
    continuous_var: str,
    categorical_vars: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute Kruskal-Wallis tests for a single continuous variable
    across multiple categorical variables in a DataFrame.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing continuous and categorical variables.
        continuous_var (str): Name of the continuous variable.
        categorical_vars (List[str], optional): List of categorical variables to test. 
            If None, all object or category dtype columns are used.

    Returns:
        pd.DataFrame: DataFrame with one row per categorical variable containing:
            - 'categorical_var': Name of the categorical variable
            - 'H_statistic': Kruskal-Wallis H-statistic
            - 'p_value': p-value from the test
            - 'n_groups': Number of groups
            - 'n_total': Total number of observations used
    """
    # Select categorical variables if not passed in
    if categorical_vars is None:
        categorical_vars = dataframe.select_dtypes(include = ['object', 'category']).columns.tolist()
    
    results = []

    for cat_var in categorical_vars:
        # Extract values of continuous variable grouped into arrays by category group
        groups = [group[continuous_var].dropna().values 
                  for name, group in dataframe.groupby(cat_var, observed = True)]

        # If there are not at least 2 groups, category is univalent
        if len(groups) < 2:
            continue

        # Calculate Kruskal-Wallis H-stat and p-value
        H_stat, p_val = kruskal(*groups)
        
        results.append({
            'categorical_var': cat_var,
            'H_statistic': H_stat,
            'p_value': p_val,
            'n_groups': len(groups),
            'n_total': sum(len(g) for g in groups)
        })
    
    return pd.DataFrame(results)
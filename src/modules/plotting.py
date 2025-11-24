from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# Custom utility functions for formatting plots
from modules.plotting_utils import (
    snake_to_title,
    snake_to_title_axes,
    snake_to_title_ticks,
)

# ---- Basic Plots ----

def custom_histplot(
    dataframe: pd.DataFrame,
    x: str,
    stat: str = "proportion",
    bins: Optional[int] = 30,
    binwidth: Optional[float] = None,
    log1p: bool = False,
    kde: bool = False,
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    xlim: Optional[tuple] = None,
    ax: Optional[Axes] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
    alpha: float = 1.0
) -> Axes:
    """
    Draw a customized histogram using seaborn with optional log1p transformation,
    KDE smoothing, and adjustable bin width.

    Args:
        dataframe (pd.DataFrame):
            Input dataframe containing the numeric variable to plot.
        x (str):
            Column name for the numeric variable to plot.
        stat (str, optional):
            Statistic for histogram ('count', 'frequency', 'proportion', etc.).
            Defaults to 'proportion'.
        bins (int, optional):
            Number of bins for the histogram. Ignored if binwidth is provided.
            Defaults to 30.
        binwidth (float, optional):
            Width of each bin. Overrides `bins` if provided. Defaults to None.
        log1p (bool, optional):
            If True, apply np.log1p transformation to x before plotting. Defaults to False.
        kde (bool, optional):
            If True, overlay a KDE curve on the histogram. Defaults to False.
        figsize (tuple, optional):
            Figure size if axis is created. Defaults to (10, 6).
        title (str, optional):
            Plot title. If None, a default title is generated. Defaults to None.
        xlabel (str, optional):
            Custom label for the x-axis. If None, the column name is used. Defaults to None.
        xlim (tuple, optional):
            Tuple specifying x-axis limits as (xmin, xmax). Defaults to None.
        ax (matplotlib.axes.Axes, optional):
            Axis to draw the plot on. If None, a new axis is created.
        color (str, optional):
            Bar color. Defaults to None.
        label (str, optional):
            Label for the histogram (for legend). Defaults to None.
        alpha (float, optional):
            Transparency of bars. Defaults to 1.0.

    Returns:
        matplotlib.axes.Axes:
            Axis containing the histogram.
    """
    # Ensure axis exists
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)

    # Prepare data
    data_to_plot = dataframe[x].copy()
    if log1p:
        data_to_plot = np.log1p(data_to_plot)

    # Draw the histogram
    sns.histplot(
        data = dataframe.assign(**{x: data_to_plot}),
        x = x,
        stat = stat,
        kde = kde,
        ax = ax,
        color = color,
        label = label,
        alpha = alpha,
        bins = None if binwidth else bins,  # use bins only if binwidth not provided
        binwidth = binwidth
    )

    # Set plot title
    if title is None:
        title_prefix = "Log1p " if log1p else ""
        title = f"{stat.title()} Histogram of {title_prefix}{snake_to_title(x)}"
    ax.set_title(title)

    # Set custom x-axis label (or use default prettified label)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        snake_to_title_axes(ax)

    # Apply x-axis limits if specified
    if xlim is not None:
        ax.set_xlim(xlim)

    # Prettify tick labels
    snake_to_title_ticks(ax, y = True)

    return ax

def custom_boxplot(
    dataframe: pd.DataFrame,
    y: str,
    hue: Optional[str] = None,
    showfliers: bool = False,
    whis: float = 1.5,
    linewidth: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None
) -> Axes:
    """
    Create a customized Seaborn boxplot with optional hue and flexible styling.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing the data to plot.
        y (str): Column name representing the variable to plot on the y-axis.
        hue (Optional[str], optional): Column name to group data by color (categorical hue). Defaults to None.
        showfliers (bool, optional): Whether to display outlier points. Defaults to False.
        whis (float, optional): Proportion of the interquartile range for whiskers. Defaults to 1.5.
        linewidth (Optional[float], optional): Width of the boxplot lines. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size if a new plot is created. Defaults to (10, 6).
        title (Optional[str], optional): Custom plot title. If None, generates one automatically. Defaults to None.
        ylabel (Optional[str], optional): Custom label for the y-axis.
            If None, uses prettified version of `y`. Defaults to None.
        ax (Optional[Axes], optional): Existing matplotlib axis to plot on. If None, a new one is created. Defaults to None.
        label (Optional[str], optional): Label for the plotted dataset (useful for legends). Defaults to None.
        color (Optional[str], optional): Base color for the boxes. Defaults to None.

    Returns:
        Axes: The matplotlib Axes object containing the resulting boxplot.
    """
    # Ensure axis exists or create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)

    # Create the boxplot
    sns.boxplot(
        data = dataframe,
        y = y,
        hue = hue,
        ax = ax,
        showfliers = showfliers,
        whis = whis,
        linewidth = linewidth,
        color = color,
        label = label
    )

    # Automatically generate a readable title if none is provided
    if title is None:
        hue_title = f" by {hue.replace('_', ' ').title()}" if hue else ""
        title = f"{y.replace('_', ' ').title()} Boxplot{hue_title}"
    ax.set_title(title, fontsize = 14)

    # Set custom or prettified axis labels
    ax.set_xlabel("")  # boxplots are vertical, x-axis label usually omitted
    if ylabel:
        ax.set_ylabel(ylabel, fontsize = 12)
    else:
        ax.set_ylabel(y.replace('_', ' ').title(), fontsize = 12)

    # Adjust label size
    ax.tick_params(axis = 'both', labelsize = 10)

    return ax

def correlation_heatmap(
    dataframe: pd.DataFrame,
    figsize: tuple[float, float] = (12, 6),
    abs: bool = True
) -> plt.Figure:
    """
    Draw a Pearson correlation heatmap with title-cased axis labels.

    Args:
        dataframe (pd.DataFrame):
            Input dataframe containing numeric columns to correlate.
        figsize (tuple, optional):
            Figure size if axis is created. Defaults to (12, 6).
        abs (bool, optional):
            If True, take the absolute value of correlations.
            If False, keep signed correlations. Defaults to True.

    Returns:
        matplotlib.figure.Figure:
            Figure containing the Pearson correlation heatmap.
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize = figsize)

    # Compute Pearson correlations
    corr = dataframe.corr(method = 'pearson', numeric_only = True)

    # Optionally take absolute value
    if abs:
        corr = corr.abs()

    # Draw heatmap
    sns.heatmap(corr, annot = True, cbar = False, ax = ax, fmt = ".2f")

    # Title
    ax.set_title("Pearson Correlation")

    # Format tick labels to title case
    snake_to_title_ticks(ax, y = True, rotation_x = 45, rotation_y = 0)

    plt.tight_layout()

    return fig

# ---- PCA Projection Plots ----

def custom_pca_plot(
    dataframe: pd.DataFrame,
    masks: Union[
        Sequence[bool], 
        np.ndarray, 
        Sequence[Union[Sequence[bool], np.ndarray]]
    ],
    labels: Optional[Union[str, Sequence[str]]] = None,
    colors: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (8, 6),
    alpha: float = 0.75,
    title: str = "PCA with Anomalies"
) -> tuple[plt.Figure, plt.Axes]:
    """
    Perform PCA on numeric columns and plot Component 1 vs Component 2.
    Supports multiple boolean masks with custom labels and colors.

    Args:
        dataframe (pd.DataFrame):
            Input dataframe containing numeric columns.
        masks (array-like or list of array-like):
            One or more boolean masks indicating subsets of points to highlight.
        labels (str or list of str, optional):
            Labels for each mask. Defaults to generic numbered labels.
        colors (list of str, optional):
            Colors for each mask. If None, defaults to the 'tab10' palette.
        ax (matplotlib.axes.Axes, optional):
            Optional axis object to draw on. If None, a new figure/axis is created.
        figsize (tuple, optional):
            Figure size when creating a new axis.
        alpha (float, optional):
            Transparency for normal points.
        title (str, optional):
            Title for the plot.

    Returns:
        (matplotlib.figure.Figure, matplotlib.axes.Axes):
            The PCA scatter plot figure and axis.
    """

    # Convert single mask to list
    if isinstance(masks, list) and not all([isinstance(item, list) for item in masks]):
        masks = [masks]

    n_masks = len(masks)

    # Handle labels
    if labels is None:
        labels = [f"Group {i + 1}" for i in range(n_masks)]
    elif isinstance(labels, str):
        labels = [labels]
    if len(labels) != n_masks:
        raise ValueError("Number of labels must match number of masks.")

    # Handle colors
    if colors is None:
        colors = sns.color_palette("tab10", n_colors = n_masks + 1)[1:]
    elif isinstance(colors, str):
        colors = [colors]
    if len(colors) != n_masks:
        raise ValueError("Number of colors must match number of masks.")

    # Extract numeric features
    X = dataframe.select_dtypes("number")

    # Fit PCA
    pca = PCA(n_components = 2, random_state = 42)
    X_pca = pca.fit_transform(X)

    # Create axis if needed
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)
    else:
        fig = ax.figure

    # Normal points = points not in ANY mask
    # combined_mask = np.unique(np.array(masks).flatten())
    sns.scatterplot(
        x = X_pca[:, 0],
        y = X_pca[:, 1],
        alpha = alpha,
        ax = ax,
        color = sns.color_palette("tab10")[0],
        label = "Normal"
    )

    # Plot each anomaly/mask group
    for mask, label, color in zip(masks, labels, colors):
        sns.scatterplot(
            x = X_pca[mask, 0],
            y = X_pca[mask, 1],
            ax = ax,
            alpha = alpha,
            color = color,
            label = label
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)

    return fig, ax

def compare_gmm_pca(
    comparison_rows: Sequence[dict],
    predictions_df: pd.DataFrame,
    test_df: pd.DataFrame,
    figsize: tuple[float, float] = (12, 30),
    sharex: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Generate PCA plots comparing GMM anomaly predictions for multiple configurations.

    Args:
        comparison_rows (list of dict):
            Each dict must contain:
                - "covariance_type": str
                - "n_components": int
            These values determine which prediction column to extract.
        predictions_df (pd.DataFrame):
            DataFrame containing prediction columns named in the format:
            'covariance_type_{cov}-n_components_{n}-predictions'
        test_df (pd.DataFrame):
            DataFrame of test samples used to generate PCA plots.
        figsize (tuple, optional):
            Figure size. Defaults to (12, 30).
        sharex (bool, optional):
            Whether subplots share the x-axis. Defaults to True.

    Returns:
        (matplotlib.figure.Figure, np.ndarray):
            The figure and array of axis objects.
    """

    # Create stacked subplots
    fig, axes = plt.subplots(
        len(comparison_rows), 1,
        figsize = figsize,
        sharex = sharex
    )

    # If only one row, axes is not an array, fix to keep interface consistent
    if len(comparison_rows) == 1:
        axes = np.array([axes])

    # Loop over GMM configs and axes
    for row_dict, ax in zip(comparison_rows, axes):
        cov_type = row_dict["covariance_type"]
        n_components = row_dict["n_components"]

        # Column name: covariance_type_{cov}-n_components_{n}-predictions
        col_name = (
            f"covariance_type_{cov_type}-"
            f"n_components_{n_components}-predictions"
        )

        if col_name not in predictions_df.columns:
            raise KeyError(f"Column '{col_name}' not found in predictions_df")

        # Predictions are typically 0/1, so extract indices
        pred_mask = predictions_df[predictions_df[col_name] == 1].index.tolist()

        # Call the PCA visualization
        custom_pca_plot(
            dataframe = test_df,
            masks = [pred_mask],
            labels = ["Predicted Anomalies"],
            ax = ax,
            title = (
                f"PCA with Predicted Anomalies "
                f"(GMM, {n_components} Components, {cov_type.title()} Covariance Type)"
            )
        )

    return fig, axes

def compare_iforest_pca(
    comparison_rows: Sequence[dict],
    predictions_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_contamination: float,
    figsize: tuple = (12, 30),
    sharex: bool = True
) -> plt.Figure:
    """
    Visualize Isolation Forest predictions on PCA-transformed test data for multiple configurations.

    Args:
        comparison_rows (Sequence[dict]):
            List of dicts with keys 'contamination_rate', 'max_features', 'max_samples'.
        predictions_df (pd.DataFrame):
            DataFrame containing prediction columns for each configuration.
        test_df (pd.DataFrame):
            Test feature DataFrame to project with PCA.
        base_contamination (float):
            Base contamination rate used to scale titles.
        figsize (tuple, optional):
            Figure size.
        sharex (bool, optional):
            Whether to share the x-axis among subplots.

    Returns:
        matplotlib.figure.Figure:
            Figure containing PCA plots with predicted anomalies.
    """

    # Create stacked subplots
    fig, axes = plt.subplots(
        len(comparison_rows), 1,
        figsize = figsize,
        sharex = sharex
    )

    # Ensure axes is always iterable
    if len(comparison_rows) == 1:
        axes = np.array([axes])

    # Loop over configurations
    for row_dict, ax in zip(comparison_rows, axes):
        con_rate = row_dict["contamination_rate"]
        max_features = float(row_dict["max_features"])
        max_samples = row_dict["max_samples"]

        # Column name containing predictions
        col_name = (
            f"contamination_{con_rate}-"
            f"max_features_{max_features}-"
            f"max_samples_{max_samples}-predictions"
        )

        if col_name not in predictions_df.columns:
            raise KeyError(f"Column '{col_name}' not found in predictions_df")

        # Index mask of predicted anomalies
        pred_mask = predictions_df[predictions_df[col_name] == 1].index.tolist()

        # PCA scatter plot with anomalies
        custom_pca_plot(
            dataframe = test_df,
            masks = [pred_mask],
            labels = ["Predicted Anomalies"],
            ax = ax,
            title = (
                f"PCA with Predicted Anomalies "
                f"(IForest, {con_rate / base_contamination:.3g}x Contamination Rate, "
                f"{max_samples} Max Samples, {max_features:.1f} Max Features)"
            )
        )

    return fig

# ---- Model Metrics and Score Histograms ----

def plot_aic_bic(
    dataframe: pd.DataFrame,
    x: str = "covariance_type",
    hue: str = "n_components",
    scores: list[str] = ["aic", "bic"],
    figsize: tuple[int, int] = (12, 6),
    suptitle: Optional[str] = "AIC and BIC Scores for Gaussian Mixture Model",
    legend_kwargs: Optional[dict] = {},
) -> plt.Figure:
    """
    Plot AIC and BIC scores for a Gaussian Mixture Model using grouped bar plots.

    Args:
        dataframe (pd.DataFrame): Input metrics table containing AIC/BIC values.
        x (str): Column to plot on the x-axis.
        hue (str): Column defining bar color groups.
        scores (list[str]): Score columns to plot (default: ['aic', 'bic']).
        figsize (tuple[int, int]): Figure size for the full plot.
        suptitle (Optional[str]): Optional figure-level title.
        legend_kwargs (Optional[dict]): Additional keyword arguments for fig.legend().

    Returns:
        matplotlib.figure.Figure: The generated matplotlib Figure.
    """
    fig, axes = plt.subplots(1, len(scores), figsize = figsize)

    # Ensure axes is iterable even if a single score is provided
    if len(scores) == 1:
        axes = [axes]

    handles, labels = None, None

    # Create each barplot for the requested score columns
    for score, ax in zip(scores, axes):
        sns.barplot(
            data = dataframe,
            x = x,
            y = score,
            hue = hue,
            ax = ax
        )

        # Extract and remove legend so we can place one shared legend later
        handles, labels = ax.get_legend_handles_labels()
        ax.legend().remove()

        ax.set_xlabel(x.replace("_", " ").title())
        ax.set_ylabel(score.upper())

    # Shared legend
    fig.legend(
        handles = handles,
        labels = labels,
        **legend_kwargs
    )

    # Optional suptitle
    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.tight_layout()
    
    return fig

def plot_gmm_metrics(
    dataframe: pd.DataFrame,
    x: str = "covariance_type",
    hue: str = "n_components",
    scores: list[str] = None,
    figsize: tuple[int, int] = (12, 6),
    suptitle: str = "Clustering Scores for Gaussian Mixture Models",
    legend_kwargs: Optional[dict] = {},
    invert_axes: Optional[dict] = None,
) -> plt.Figure:
    """
    Plot clustering evaluation metrics (e.g., silhouette, Calinski-Harabasz,
    Davies-Bouldin) for Gaussian Mixture Models using grouped bar plots.

    Args:
        dataframe (pd.DataFrame): Metrics table containing clustering scores.
        x (str): Column to plot on the x-axis (default: 'covariance_type').
        hue (str): Column defining bar color groups (default: 'n_components').
        scores (list[str], optional): List of metrics to plot. Defaults to the
            three common clustering metrics.
        figsize (tuple[int, int]): Figure size.
        suptitle (str): Suptitle for the full figure.
        legend_kwargs (dict, optional): Additional kwargs passed to fig.legend().
        invert_axes (dict, optional): Mapping from score name to True/False.
            Example: {'davies_bouldin_score': True}

            Default behavior:
                - davies_bouldin_score: inverted (lower is better)
                - others: normal

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    # Default metrics
    if scores is None:
        scores = [
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score",
        ]

    # Default inversion behavior
    default_inversions = {
        "silhouette_score": False,
        "calinski_harabasz_score": False,
        "davies_bouldin_score": True, # lower is better
    }

    # Override defaults if user provided custom inversion map
    if invert_axes is None:
        invert_axes = default_inversions
    else:
        # Apply defaults, then overwrite with user values
        merged = default_inversions.copy()
        merged.update(invert_axes)
        invert_axes = merged

    fig, axes = plt.subplots(1, len(scores), figsize = figsize)

    # Ensure axes is iterable even if there is only one score
    if len(scores) == 1:
        axes = [axes]

    handles, labels = None, None

    # Create each subplot
    for score, ax in zip(scores, axes):
        score_str = score.replace("_", " ").title()

        sns.barplot(
            data = dataframe,
            x = x,
            y = score,
            hue = hue,
            ax = ax
        )

        # Extract and remove legend so we can place one shared legend later
        handles, labels = ax.get_legend_handles_labels()
        ax.legend().remove()

        ax.set_xlabel(x.replace("_", " ").title())
        ax.set_ylabel(score_str)

        # Apply metric-specific axis inversion
        if invert_axes.get(score, False):
            ax.invert_yaxis()

    # Shared legend
    fig.legend(
        handles = handles,
        labels = labels,
        **legend_kwargs
    )

    fig.suptitle(suptitle)
    fig.tight_layout()

    return fig

def plot_iforest_metrics(
    dataframe: pd.DataFrame,
    scores: list[str] = None,
    figsize: tuple = (12, 20),
    suptitle: str = "Clustering Scores for Isolation Forest Models",
    inverse_axes: dict = None
) -> plt.Figure:
    """
    Create a stacked subfigure layout comparing clustering metrics for Isolation Forest
    models across different contamination rates, with optional y-axis inversion.

    Args:
        dataframe (pd.DataFrame):
            DataFrame containing metric results. Must include:
            'contamination_rate', 'max_samples', 'max_features', and metric columns.
        scores (list of str, optional):
            Metric column names to plot. Defaults to silhouette, calinski_harabasz, and davies_bouldin.
        figsize (tuple, optional):
            Size of the overall figure.
        suptitle (str, optional):
            Title for the full figure.
        inverse_axes (dict, optional):
            Dictionary mapping metric names to bool, indicating if axis should be inverted.
            Example: {'davies_bouldin_score': True}

    Returns:
        matplotlib.figure.Figure:
            The figure containing all comparison plots.
    """

    # Default metrics
    if scores is None:
        scores = [
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score"
        ]

    # Default axis inversion: lower is better for Davies-Bouldin
    if inverse_axes is None:
        inverse_axes = {"davies_bouldin_score": True}

    # Extract contamination rates for row plots
    contamination_rates = np.sort(dataframe.contamination_rate.unique())

    # Create general figre
    fig = plt.figure(constrained_layout = True, figsize = figsize)
    fig.suptitle(suptitle, fontsize = 16)

    # Create subfigures as rows
    subfigs = fig.subfigures(len(contamination_rates), 1)

    for i, subfig in enumerate(subfigs):
        # Select the appropriate contamination rate and subset the dataframe
        contamination = contamination_rates[i]
        sub = dataframe[dataframe.contamination_rate == contamination]

        # Set row suptitle
        scale_factor = 2 ** (-4 + i)
        subfig.suptitle(
            f"Contamination = {contamination}, {scale_factor}x Base Contamination"
        )

        # Create subplots for the row
        axes = subfig.subplots(1, len(scores))

        for j, ax in enumerate(axes):
            # For each column, plot the appropriate score
            score_col = scores[j]
            score_label = score_col.replace("_", " ").title()

            sns.barplot(
                data = sub,
                x = "max_samples",
                y = score_col,
                hue = "max_features",
                ax = ax
            )

            # Extract handles and labels, remove local legends to use shared global legend later
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

            ax.set_xlabel("Max Samples")
            ax.set_ylabel(score_label)

            # Invert axis if specified
            if inverse_axes.get(score_col, False):
                ax.invert_yaxis()

    # Shared legend at bottom center
    fig.legend(
        handles = handles,
        labels = labels,
        bbox_to_anchor = (0.525, 0),
        loc = "upper center",
        ncols = 2,
        title = "Max Features"
    )

    return fig

def score_histogram(
    dataframe: pd.DataFrame,
    score_col: str,
    contamination: float,
    bins: int = 2500,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12, 6),
    legend_kwargs: Optional[dict] = None,
    hist_color: Optional[str] = None,
    hist_label: Optional[str] = None,
    show_threshold: bool = True,
    alpha: float = 1.0,
) -> plt.Axes:
    """
    Plot a histogram of anomaly scores with optional contamination threshold line.

    Args:
        dataframe (pd.DataFrame):
            DataFrame containing a score column.
        score_col (str):
            Column name containing the anomaly scores.
        contamination (float):
            Expected anomaly rate used to compute threshold via quantile.
        bins (int, optional):
            Number of histogram bins. Defaults to 2500.
        ax (matplotlib.axes.Axes, optional):
            Optional axis to draw on. If None, a new figure & axis are created.
        title (str, optional):
            Plot title. If None, a default based on score_col is used.
        figsize (tuple, optional):
            Only used when creating a new figure. Defaults to (12, 6).
        legend_kwargs (dict, optional):
            Additional kwargs forwarded to `ax.legend()`.
        hist_color (str, optional):
            Color for the histogram bars.
        hist_label (str, optional):
            Label for the histogram (used in legend).
        show_threshold (bool, optional):
            If True, draws a vertical threshold line based on contamination.
        alpha (float, optional):
            Transparency for the histogram bars. Defaults to 1.0.

    Returns:
        matplotlib.axes.Axes:
            The axis containing the histogram.
    """

    # Ensure axis exists
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)

    scores = dataframe[score_col]

    # Compute range limits
    x_min = scores.quantile(0.0001)
    x_max = scores.max()

    # Plot histogram through custom_histplot
    custom_histplot(
        dataframe = dataframe,
        x = score_col,
        xlim = [x_min, x_max],
        bins = bins,
        title = title or f"Score Histogram ({score_col})",
        xlabel = "Scores",
        ax = ax,
        color = hist_color,
        label = hist_label,
        alpha = alpha,
    )

    # Draw contamination threshold line only if enabled
    if show_threshold:
        thresh = scores.quantile(contamination)
        ax.axvline(
            thresh,
            color = "black",
            linestyle = "--",
            label = f"Prediction Threshold = {thresh:.04f}",
        )

    # Finalize legend
    legend_kwargs = legend_kwargs or {}
    ax.legend(**legend_kwargs)

    return ax
import math

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, Optional

def dist_visualisation(
    data: Union[np.ndarray, pd.Series, list],
    title: str = "",
    show: bool = True
) -> Optional[Tuple[plt.Figure, Dict[str, plt.Axes]]]:
    """
    Visualize the distribution of data using boxplot, violinplot, and histogram.

    Args:
        data: Input numerical data to visualize.
        title: Title of the entire figure.
        show: If True, displays the plot. If False, returns the figure and axes.

    Returns:
        Tuple of matplotlib figure and axes dictionary if show=False, otherwise None.
    """
    fig = plt.figure(constrained_layout=True)
    plt.style.use('seaborn-v0_8')

    ax_dict = fig.subplot_mosaic([
        ['boxplot', 'violin'],
        ['histogram', 'violin'],
        ['histogram', 'violin'],
    ])
    
    mean_props = dict(color="darkorange", linestyle="dashed")

    ax_dict['boxplot'].boxplot(
        data,
        showmeans=True,
        meanline=True,
        meanprops=mean_props,
        vert=False
    )
    
    violin = ax_dict['violin'].violinplot(
        data,
        showmeans=True,
        showmedians=True
    )
    violin["cbars"].set_linestyle('dotted')

    ax_dict['histogram'].hist(data, bins='sqrt', color='orange')

    ax_dict['violin'].set_xlabel('KDE')
    ax_dict['histogram'].set_ylabel('Count Frequency')

    fig.suptitle(title)

    if show:
        plt.show()
        return None
    else:
        return fig, ax_dict


def plot_fraud_heatmap(df: pd.DataFrame, row: str, col: str, title: str = "Fraud Rate Heatmap") -> None:
    """
    Plot a heatmap of fraud rate based on two categorical variables.

    Args:
        df: DataFrame containing the data.
        row: Column to use as rows in heatmap.
        col: Column to use as columns in heatmap.
        title: Title of the heatmap.
    """
    pivot = pd.crosstab(df[row], df[col], values=df["is_fraud"], aggfunc="mean")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2%", cmap="Reds")
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel(row)
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(
    df: pd.DataFrame,
    categorical_cols: list[str],
    n_cols: int = 2,
    figsize: Optional[tuple] = None,
    title: Optional[str] = "Categorical Feature Distributions",
    save_path: Optional[str] = None
) -> None:
    """
    Plots count distributions for multiple categorical columns using subplots, in a clean and professional style.

    Args:
        df: DataFrame containing the data.
        categorical_cols: List of column names to visualize.
        n_cols: Number of columns in the subplot grid.
        figsize: Size of the entire figure. If None, inferred automatically.
        title: Title of the whole figure.
        save_path: If provided, saves the figure to this path instead of displaying it.
    """
    sns.set_style("whitegrid")  # Profesjonalny jasny styl

    n = len(categorical_cols)
    n_rows = math.ceil(n / n_cols)
    figsize = figsize or (6 * n_cols, 4.5 * n_rows)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(f"Distribution of {col}", fontsize=13, weight="bold")
        ax.set_xlabel("Count", fontsize=11)
        ax.set_ylabel(col, fontsize=11)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)
        max_val = df[col].value_counts().max()
        ax.set_xlim(0, max_val * 1.1)

    # Turn off any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if title:
        fig.suptitle(title, fontsize=16, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



def plot_correlation_matrix(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    title: str = "Correlation Matrix: Numerical Features",
    figsize: tuple = (8, 6),
    cmap: str = "coolwarm"
) -> None:
    """
    Plot a correlation matrix heatmap for selected numeric features.

    Args:
        df: DataFrame containing the data.
        cols: Optional list of numeric column names. If None, use all numeric columns.
        title: Title of the heatmap.
        figsize: Size of the matplotlib figure.
        cmap: Colormap to use for the heatmap.
    """
    sns.set_style("whitegrid")

    # Select only numeric columns
    if cols is None:
        corr_df = df.select_dtypes(include="number")
    else:
        corr_df = df[cols]

    corr_matrix = corr_df.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title, fontsize=14, weight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
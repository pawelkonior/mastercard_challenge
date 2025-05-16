
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


def plot_correlation_matrix(df: pd.DataFrame, cols: Optional[list] = None, title: str = "Correlation Matrix") -> None:
    """
    Plot a heatmap of correlation between selected numeric features.

    Args:
        df: DataFrame containing the data.
        cols: List of columns to compute correlation. If None, all numeric columns are used.
        title: Title of the plot.
    """
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def boxplot_with_swarm(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    title: str = ""
) -> None:
    """
    Draw a boxplot with an overlaid swarmplot.

    Args:
        df: DataFrame containing the data.
        x: Column name for x-axis (categorical).
        y: Column name for y-axis (numerical).
        hue: Optional column name for hue separation.
        title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y, hue=hue, palette="Set3", dodge=True)
    sns.swarmplot(data=df, x=x, y=y, hue=hue, dodge=True, color=".25", alpha=0.6)
    plt.title(title)
    plt.tight_layout()
    plt.show()

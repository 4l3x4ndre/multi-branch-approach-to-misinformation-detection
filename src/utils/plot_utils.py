"""
Scripts based on BeautifilFigures from Andrey Churkin https://andreychurkin.ru/
"""

import matplotlib.pyplot as plt
import numpy as np

# Defining colours for datasets
DATASET_COLORS = ['#95BB63', '#BCBCE0', '#77b5b6', '#EA805D']
DATASET_LINE_COLORS = ['#95BB63', '#BCBCE0', '#77b5b6', '#EA805D']
# DATASET_LINE_COLORS = ['#6a408d', '#4e4e4e', '#378d94', '#c04a2e']
REGRESSION_COLOR = '#8a8a8a'

def set_plot_style():
    """Sets a consistent style for matplotlib plots."""
    plt.rcParams.update({
        'font.family': 'monospace',
        # 'font.family': 'Courier New',
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 20,
        # 'svg.fonttype': 'Courier New',
        'pdf.fonttype': 42,              # Embed fonts as TrueType (keeps text selectable)
        'ps.fonttype': 42,               # For PostScript 
    })

def get_styled_figure_ax(figsize=(10, 10), aspect='equal', datalim=True, grid=True):
    """
    Creates a matplotlib figure and axes with a consistent style.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    if aspect == 'equal':
        ax.set_aspect('equal', adjustable='datalim' if datalim else 'box')

    if grid:
        # Major grid
        ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25)

        # Minor ticks and grid
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.15)

        ax.set_axisbelow(True)
    return fig, ax

def convert_label(label):
    """Converts internal label names to more readable forms."""
    if label == 'claim_real':
        return 'Claim Real'
    elif label == 'img_real':
        return 'Image Real'
    elif label == 'mismatch':
        return 'Mismatch'
    elif label == 'overall':
        return 'Overall Truth'
    else:
        return label

def style_legend(ax, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=4, frameon=False, order=None):
    """Styles the legend of a plot."""
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        if order:
            handles = [handles[i] for i in order]
            labels = []
            for i in order:
                labels.append(convert_label(labels[i]))
        else:
            labels = [convert_label(label) for label in labels]
        ax.legend(
            handles,
            labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=ncol,
            frameon=frameon
        )

def adjust_plot_limits(ax, all_data_x, all_data_y, zoom_out=0.6):
    """Adjusts plot limits to be equal and centered around the data."""
    if not hasattr(all_data_x, '__len__') or len(all_data_x) == 0 or not hasattr(all_data_y, '__len__') or len(all_data_y) == 0:
        return

    x_min = min(all_data_x)
    x_max = max(all_data_x)
    x_median = (x_min + x_max) / 2
    x_range = x_max - x_min

    y_min = min(all_data_y)
    y_max = max(all_data_y)
    y_median = (y_min + y_max) / 2
    y_range = y_max - y_min

    plotting_range = max([x_range, y_range]) + zoom_out

    ax.set_xlim(x_median - plotting_range / 2, x_median + plotting_range / 2)
    ax.set_ylim(y_median - plotting_range / 2, y_median + plotting_range / 2)

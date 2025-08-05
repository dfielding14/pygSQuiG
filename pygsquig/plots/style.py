"""Plot styling configuration for pygSQuiG visualizations."""

import matplotlib.pyplot as plt


class PlotStyle:
    """Consistent plot styling for publication quality figures."""

    # Color schemes
    FIELD_CMAP = "RdBu_r"
    VORTICITY_CMAP = "seismic"
    SPECTRUM_COLOR = "#1f77b4"
    REFERENCE_COLOR = "#ff7f0e"

    # Figure defaults
    DPI = 150
    FIGSIZE_SINGLE = (6, 5)
    FIGSIZE_DOUBLE = (12, 5)
    FIGSIZE_GRID = (12, 10)

    @staticmethod
    def setup():
        """Set up matplotlib parameters for consistent styling."""
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.labelsize": 14,
                "axes.titlesize": 16,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "figure.titlesize": 18,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "lines.linewidth": 2,
            }
        )


# Initialize plot style when module is imported
PlotStyle.setup()

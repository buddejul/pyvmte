"""Functions for plotting results of monte carlo simulations."""

import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore


def plot_upper_and_lower_bounds(results: dict) -> go.Figure:
    """Returns plot of upper and lower bounds histograms."""

    upper_bounds = results["upper_bound"]
    lower_bounds = results["lower_bound"]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=upper_bounds, histnorm="probability", name="Upper Bounds", nbinsx=20
        )
    )
    fig.add_trace(
        go.Histogram(
            x=lower_bounds, histnorm="probability", name="Lower Bounds", nbinsx=20
        )
    )

    fig.update_layout(
        title_text="Upper and Lower Bounds",
        xaxis_title_text="Bounds",
        yaxis_title_text="Frequency",
    )

    return fig

import plotly.express as px
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.gofplots import qqplot
from utils.fcf_analysis.noisefuncs import identify_outlier_groups
matplotlib.use("agg")

margin_defaults = {
    "b": 10,
    "l": 5,
    "r": 5,
    "t": 10,
}


def plot_blank_scatterplot(fig_text="Select point to view time series"):
    fig = px.scatter()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": fig_text,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 28
                    }
                }
            ],
        margin=margin_defaults
        )
    return fig


def plot_time_series(
        ts,
        dates,
        # pid: str,
        # cls: str,
        # prob: float
):
    """Plotly plot of the EGMS time series data"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ts,
            connectgaps=True,
            mode="lines+markers",
            line=dict(color="firebrick", width=1, dash="dot"),
            hovertemplate="%{y:.2f}mm, %{x|%Y-%m-%d}<br><extra></extra>",
            showlegend=False
        )
    )
    fig.update_layout(
            # title=dict(
            #     text=(
            #         f"Time series plot for PID: <b>{pid}</b><br>"
            #         f"Classified trend type: <b style='color:black;'>{cls}</b><br>"
            #         f"Class probability: <b style='color:black;'>{prob:.2f}</b>"
            #     )
            # ),
            xaxis=dict(
                title=dict(
                    text="Date"
                )
            ),
            yaxis=dict(
                title=dict(
                    text="Displacement (mm)"
                )
            ),
            margin=margin_defaults
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def plot_time_series_decomp(
        ts,
        trend,
        seasonality,
        dates,
        # pid: str,
        # cls: str,
        # prob: float
):
    """Plotly plot of the classified EGMS time series data"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ts,
            connectgaps=True,
            mode="lines+markers",
            line=dict(color="firebrick", width=1, dash="dot"),
            hovertemplate="%{y:.2f}mm, %{x|%Y-%m-%d}<br><extra></extra>",
            name="Raw data",
            showlegend=True
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=trend,
            connectgaps=True,
            mode="lines",
            line=dict(color="blue", width=1.5),
            hovertemplate="%{y:.2f}mm, %{x|%Y-%m-%d}<br><extra></extra>",
            name="Trend",
            showlegend=True
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=seasonality,
            connectgaps=True,
            mode="lines",
            line=dict(color="orange", width=1.5),
            hovertemplate="%{y:.2f}mm, %{x|%Y-%m-%d}<br><extra></extra>",
            name="Seasonality",
            showlegend=True
        )
    )
    fig.update_layout(
        # title=dict(
        #     text=(
        #         f"Time series plot for PID: <b>{pid}</b><br>"
        #         f"Classified trend type: <b style='color:black;'>{cls}</b><br>"
        #         f"Class probability: <b style='color:black;'>{prob:.2f}</b>"
        #     )
        # ),
        xaxis=dict(
            title=dict(
                text="Date"
            )
        ),
        yaxis=dict(
            title=dict(
                text="Displacement (mm)"
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        margin=margin_defaults
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def plot_time_series_residuals(
        resids,
        dates
):
    """Plotly plot of the classified EGMS time series data"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=resids,
            connectgaps=True,
            mode="markers",
            hovertemplate="%{y:.2f}mm, %{x|%Y-%m-%d}<br><extra></extra>",
            name="Residuals",
            showlegend=False
        )
    )
    fig.add_hline(
        y=0, line_width=1, line_dash="dot",
    )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Date"
            )
        ),
        yaxis=dict(
            title=dict(
                text="Residuals (mm)"
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=margin_defaults
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def plot_fitted_residuals(
        resids,
        fitted
):
    """Plotly plot of the classified EGMS time series data"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=resids,
            connectgaps=True,
            mode="markers",
            hovertemplate="%{y:.2f}mm, %{x:.2f}mm<br><extra></extra>",
            name="Residuals",
            showlegend=False
        )
    )
    fig.add_hline(
        y=0, line_width=1, line_dash="dot",
    )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Fitted values (mm)"
            )
        ),
        yaxis=dict(
            title=dict(
                text="Residuals (mm)"
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=margin_defaults
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def plot_qq(resids):
    """Plotly QQ plot of residuals - test for normality
    from https://plotly.com/python/v3/normality-test/
    """
    qqplot_data = qqplot(resids, line="s").gca().lines
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=qqplot_data[0].get_xdata(),
            y=qqplot_data[0].get_ydata(),
            mode="markers",
            marker={"color": "#19d3f3"}
        )
    )
    fig.add_trace(
        go.Scatter(
            x=qqplot_data[1].get_xdata(),
            y=qqplot_data[1].get_ydata(),
            mode="lines",
            line=dict(color="blue", width=1.5),
        )
    )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Theoritical Quantities",
            )
        ),
        yaxis=dict(
            title=dict(
                text="Sample Quantities"
            )
        ),
        showlegend=False,
        margin=margin_defaults
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def plot_psd(psd, freqs, psd_thr, freqs_thr):
    """Plot the power spectrum densit (PSD) for the
    sample frequencies"""

    psd_colors = [["lightslategray", "crimson"][p >= psd_thr] for p in psd]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=freqs,
            y=psd,
            marker_color=psd_colors,
            hovertemplate="Ann. freq: %{x:.4f}<br>PSD: %{y:.4f}<br><extra></extra>",
        )
    )
    fig.add_hline(y=psd_thr, line_width=1, line_dash="dot",)
    fig.add_vrect(
        x0=freqs_thr[0], x1=freqs_thr[1], 
        annotation_text="Frequency range", annotation_position="top left",
        fillcolor="green", opacity=0.25, line_width=0
    )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Annual Frequency",
            )
        ),
        yaxis=dict(
            title=dict(
                text="PSD"
            )
        ),
        showlegend=False,
        margin=margin_defaults
    )
    return fig


def plot_seasonality_ts(seasonal_ts, dates, peaks, troughs):
    """Plotly QQ plot of residuals - test for normality
    from https://plotly.com/python/v3/normality-test/
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=seasonal_ts,
            mode="lines",
            line=dict(color="orange", width=1.5),
            hovertemplate="%{y:.2f}mm, %{x|%Y-%m-%d}<br><extra></extra>",
        )
    )
    # Don't index by empty list if no seasonality
    if sum(abs(seasonal_ts)) > 0:
        fig.add_trace(
            go.Scatter(
                x=dates[troughs],
                y=seasonal_ts[troughs],
                mode="markers",
                marker={"color": "#19d3f3"},
                hovertemplate="%{y:.2f}mm, %{x|%Y-%m-%d}<br><extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates[peaks],
                y=seasonal_ts[peaks],
                mode="markers",
                marker={"color": "#19d3f3"},
                hovertemplate="%{y:.2f}mm, %{x|%Y-%m-%d}<br><extra></extra>",
            )
        )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Date",
            )
        ),
        yaxis=dict(
            title=dict(
                text="Displacement (mm)"
            )
        ),
        showlegend=False,
        margin=margin_defaults
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def get_marker_colors_from_thr(
        x: np.ndarray, thr: float, colors: list
) -> list:
    return np.where(x < thr, colors[0], colors[1]).tolist()


def plot_ts_outliers(
        ts: np.ndarray, dates: list, thr: float,
        wsize: int = 12, n_outliers: int = 6,
        colors=["blue", "firebrick"]
):
    marker_colors = np.where(np.abs(np.diff(ts)) > thr, colors[1], colors[0])
    marker_colors = np.concatenate([[colors[0]], marker_colors])
    marker_symbols = np.where(np.abs(np.diff(ts)) > thr, 4, 0)
    marker_symbols = np.concatenate([[0], marker_symbols])
    outlier_groups = identify_outlier_groups(ts, thr, wsize, n_outliers)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ts,
            connectgaps=True,
            mode="lines+markers",
            line=dict(color=colors[0], width=1, dash="dot"),
            marker=dict(color=marker_colors),
            marker_symbol=marker_symbols,
            hovertemplate="%{y:.2f}mm, %{x|%Y-%m-%d}<br><extra></extra>",
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[None],
            mode="markers",
            marker=dict(symbol=4, color=colors[1]),
            hovertemplate="<extra></extra>",
            name="Outliers",
            showlegend=True
        )
    )
    if len(outlier_groups) > 0:
        for idxs in outlier_groups:
            fig.add_vrect(
                x0=dates[idxs[0]], x1=dates[idxs[1]],
                line_width=0, fillcolor="red", opacity=0.2,
                name="Noise burst", showlegend=True,
            )
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Date"
            )
        ),
        yaxis=dict(
            title=dict(
                text="Displacement (mm)"
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig

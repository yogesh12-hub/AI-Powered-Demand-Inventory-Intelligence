"""Plotly charts with the FORESIGHT dark theme. Every chart takes real data frames; none invents values."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

NAVY, CARD, GRID, TEXT, MUTED = "#0b1220", "#111a2e", "#1f2a44", "#e6edf7", "#8b9ab5"
ORANGE, GREEN, RED, AMBER, BLUE = "#ff7a1a", "#22c55e", "#ef4444", "#f59e0b", "#5b9dff"
RISK_COLORS = {"High": RED, "Medium": AMBER, "Low": GREEN}
PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": "foresight_chart", "scale": 2},
               "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def theme(fig: go.Figure, height: int = 400, title: str | None = None) -> go.Figure:
    """Shared chart style: 14px text, title above a separate legend row, generous margins, readable dates."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=16, r=16, t=(86 if title else 52), b=16),
                      font=dict(color=TEXT, size=14), hovermode="x unified",
                      title=dict(text=title, font=dict(size=17), x=0, xanchor="left", y=0.97, yanchor="top") if title else None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=13), bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(font_size=14, bgcolor=CARD, bordercolor=GRID))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=13), title_font=dict(size=14), automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=13), title_font=dict(size=14), automargin=True)
    return fig


def _dates(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(tickformat="%b %Y", hoverformat="%d %b %Y", nticks=6, tickangle=0)
    return fig


def sales_trend(df: pd.DataFrame, title="Units sold and revenue") -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=df["date"], y=df["units_sold"], name="Units sold", marker_color=ORANGE, opacity=0.85)
    if df["revenue"].notna().any():
        fig.add_scatter(x=df["date"], y=df["revenue"], name="Revenue (₹)", yaxis="y2", line=dict(color=BLUE, width=2))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, tickprefix="₹"))
    return _dates(theme(fig, 420, title))


def bar(df, x, y, title, color=ORANGE, horizontal=False, height=400, fmt=None) -> go.Figure:
    fig = go.Figure(go.Bar(x=df[y] if horizontal else df[x], y=df[x] if horizontal else df[y],
                           orientation="h" if horizontal else "v", marker_color=color))
    if horizontal:
        fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    return theme(fig, height, title)


def seasonality_bar(mi: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=mi["month_name"], y=mi["index"], marker_color=[GREEN if v >= 1 else AMBER for v in mi["index"]]))
    fig.add_hline(y=1, line_dash="dash", line_color=MUTED)
    fig.update_layout(hovermode="closest")
    return theme(fig, 380, "Seasonal index by month (1.00 = average month)")


def forecast_chart(history: pd.Series, fc: pd.DataFrame, show_intervals=True, show_baseline=True, tail_weeks=52, title="History vs forecast") -> go.Figure:
    fig = go.Figure()
    h = history.tail(tail_weeks)
    fig.add_scatter(x=h.index, y=h.values, name="Actual", line=dict(color=TEXT, width=2))
    if show_intervals and fc["lower"].notna().any():
        fig.add_scatter(x=fc["week_start"], y=fc["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fc["week_start"], y=fc["lower"], fill="tonexty", fillcolor="rgba(255,122,26,0.18)", line=dict(width=0), name="Prediction interval")
    if show_baseline:
        fig.add_scatter(x=fc["week_start"], y=fc["baseline"], name="Seasonal-naive baseline", line=dict(color=BLUE, dash="dot"))
    fig.add_scatter(x=fc["week_start"], y=fc["forecast"], name="Forecast", line=dict(color=ORANGE, width=3))
    return _dates(theme(fig, 480, title))


def backtest_chart(bt: pd.DataFrame, model: str) -> go.Figure:
    """Aggregate actual vs forecast by week across evaluated SKUs (model and baseline)."""
    m = bt[bt.model == model].groupby("week_start")[["actual", "forecast"]].sum()
    b = bt[bt.model == "seasonal_naive"].groupby("week_start")["forecast"].sum()
    fig = go.Figure()
    fig.add_scatter(x=m.index, y=m["actual"], name="Actual", line=dict(color=TEXT, width=2))
    fig.add_scatter(x=m.index, y=m["forecast"], name=f"{model}", line=dict(color=ORANGE, width=2))
    if model != "seasonal_naive":
        fig.add_scatter(x=b.index, y=b.values, name="seasonal_naive", line=dict(color=BLUE, dash="dot"))
    return _dates(theme(fig, 400, "Backtest: actual vs predicted (all evaluated SKUs, held-out weeks)"))


def risk_heatmap(view: pd.DataFrame) -> go.Figure:
    v = view.copy()
    v["category"] = v["category"].fillna("(uncategorised)")
    cols = {"Stockout<br>High": ("stockout_risk", "High"), "Stockout<br>Medium": ("stockout_risk", "Medium"),
            "Overstock<br>Medium": ("overstock_risk", "Medium"), "Overstock<br>High": ("overstock_risk", "High")}
    z = pd.DataFrame({k: v[c].eq(val).groupby(v["category"]).sum() for k, (c, val) in cols.items()})
    fig = go.Figure(go.Heatmap(z=z.values, x=list(z.columns), y=list(z.index), colorscale=[[0, CARD], [1, ORANGE]],
                               text=z.values, texttemplate="%{text}", hovertemplate="%{y} / %{x}: %{z} SKUs<extra></extra>"))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(tickangle=0, side="bottom")
    fig.update_traces(textfont_size=15, colorbar=dict(thickness=12))
    return theme(fig, max(340, 150 + 48 * len(z)), "SKU count by category and risk")


def coverage_hist(view: pd.DataFrame, cap=365) -> go.Figure:
    c = view["coverage_days"].replace(np.inf, np.nan).dropna().clip(upper=cap)
    fig = go.Figure(go.Histogram(x=c, nbinsx=30, marker_color=ORANGE))
    fig.update_layout(hovermode="closest", xaxis_title=f"Stock coverage (days, capped at {cap})", yaxis_title="SKUs")
    return theme(fig, 380, "Distribution of stock coverage")


def risk_donut(counts: pd.Series, title: str) -> go.Figure:
    fig = go.Figure(go.Pie(labels=list(counts.index), values=list(counts.values), hole=0.62,
                           marker_colors=[RISK_COLORS[k] for k in counts.index], textinfo="value", sort=False))
    fig.update_traces(textfont_size=15)
    theme(fig, 360, title)
    fig.update_layout(hovermode="closest", showlegend=True, legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02, yanchor="bottom"))
    return fig

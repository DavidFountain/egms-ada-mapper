import dash_bootstrap_components as dbc
from dash import html, Input, Output
from assets.style import SIDEBAR_STYLE

# from docs https://dash-bootstrap-components.opensource.faculty.ai/docs/components/navbar/

sidebar = html.Div(
    [
        html.H2("EGMS Data Viewer", className="display-5"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Load Data", href="/", active="exact"),
                dbc.NavLink("Analysis", href="/analysis", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

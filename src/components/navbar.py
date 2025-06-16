from dash import html
import dash_bootstrap_components as dbc
from assets.style import NAVBAR_STYLE

# from docs: https://dash-bootstrap-components.opensource.faculty.ai/docs/components/navbar/
navbar_simple = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("", href="/")),
    ],
    brand="EGMS Time Series Classification",
    brand_href="#",
    color=NAVBAR_STYLE["background-color"],
    dark=False,
    fluid=False
)


navbar = dbc.Navbar(
    [
        dbc.Row(
            style={"width": "100%"},
            align="center",
            children=[
                dbc.Col(width=3, style={"display": "flex", "align-items": "left"},
                        children = [
                            html.A(
                                dbc.NavbarBrand(
                                    "EGMS Time Series Classification",
                                    class_name="ml-2",
                                    style={
                                        "padding-top": "160px",
                                        "padding-left": "2rem",
                                        "font-size": "2em",
                                        # "font-weight": "bold",
                                    }
                                )
                            )
                        ]
                )
            ]
        )
    ],
    color=NAVBAR_STYLE["background-color"],
)







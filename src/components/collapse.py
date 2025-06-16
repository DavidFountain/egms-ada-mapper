from dash import html
import dash_bootstrap_components as dbc


def render_collapse(
        collapse_body, id: str, title: str, class_name: str="d-grid gap-2"):
    collapse = html.Div(
        [
            dbc.Button(
                title,
                id=f"{id}-button",
                className="me-1",
                color="light",
                n_clicks=0,
            ),
            dbc.Collapse(
                collapse_body,
                id=id,
                is_open=False,
            ),
        ],
        className=class_name,
    )
    return collapse

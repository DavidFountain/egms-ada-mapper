import dash_bootstrap_components as dbc


def render_input_group(id: str, title: str, placeholder: str, **kwargs):
    input_group = dbc.InputGroup(
        [
            dbc.InputGroupText(title),
            dbc.Input(id=id, placeholder=placeholder, **kwargs)
        ],
        className="mb-3"
    )
    return input_group

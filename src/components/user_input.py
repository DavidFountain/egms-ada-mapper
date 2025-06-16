from dash import dcc


def render_number_input(id: str, items: list=[""],
                    placeholder: str="Input number here..."):
    inpt = dcc.Input(
        id=id,
        type="number",
        placeholder=placeholder,
        value=items[0],
    )
    return inpt

from dash import dcc


def prettify_string(string):
    """Replace non-alphanumeric characters
    and capitalise strings for graphs"""
    return string.replace("_", " ").capitalize()


def render_dropdown(
        id: str, items: list | dict,
        clearable_option: bool = False,
        default: str | None = None):
    if isinstance(items, list):
        options_ = [
            {'label': prettify_string(item)
             if isinstance(item, str) else item, 'value': item}
            for item in items
        ]
        default_ = items[0]
    else:
        options_ = [
            {'label': key, 'value': val}
            for key, val in items.items()
        ]
        default_ = list(items.values())[0]

    dropdown = dcc.Dropdown(
        id=id,
        clearable=clearable_option,
        options=options_,
        value=default_ if default is None else default,
        style={
            "font-size": "90%",
            "width": "100%",
            "display": "inline-block",
        }
    )
    return dropdown

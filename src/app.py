import os
import re
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import urllib.request
import uuid
from datetime import datetime

import dash
from dash import Dash, callback, Input, Output, State, html, dcc, dash_table
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import arrow_function, assign
from flask import Flask

from config import (ADA_CLASS_DIR, AOI_BOUNDS_DIR, PID_FILE_LU_DIR,
                    aoi_country_lookup, aoi_geohaz_lookup, aoi_name_lookup)
from components.dropdown import render_dropdown
import utils.fcf_analysis.fourier_trend_analysis as fa
import utils.fcf_analysis.insartsclassifier as incls

from utils.dataprocessing import (prettify_egms_date, get_date_cols,
                                  resample_egms_data, get_most_common_sr,
                                  convert_json_to_dataframe,
                                  create_velocity_groups
                                  )

from utils.dataio import (get_ada_location, get_aoi_bounds_location,
                          get_pid_lookup_location, read_geo_data,
                          get_ts_filename, read_single_ts)

from visualisations.cmaps import (adatype_cmap, adasubtype_cmap,
                                  trend_class_cmap, trend_subclass_cmap,
                                  insar_velocity_colors, label_prob_colors,
                                  stable_prop_colors, metric_color_dict,
                                  insar_vel_grp_cmap)

from visualisations.visualisations import (plot_time_series_decomp,
                                           plot_time_series_residuals,
                                           plot_qq,
                                           plot_psd,
                                           plot_seasonality_ts,
                                           plot_fitted_residuals,
                                           plot_ts_outliers,
                                           plot_blank_scatterplot)


gpd.options.io_engine = "pyogrio"
os.environ["PYOGRIO_USE_ARROW"] = "1"

chroma = "https://cdnjs.cloudflare.com/ajax/libs/chroma-js/2.1.0/chroma.min.js"   # js lib used for colors
LOGO = "https://placehold.co/100x100"

# -------------------------------------------------
# ---------- INITIAL VALUES FOR DISPLAY -----------
# -------------------------------------------------

INIT_AOI = "mans-chesterf-notts_27700"
INIT_AOI_CLEAN = re.sub(r"^(.*)_\d+$", "\\1", INIT_AOI)
INIT_MODEL_DATE = "20250608"
INIT_EGMS_DATE = "20182022"
INIT_PRODUCT = "basic"
INIT_S1PATH = "asc"
INIT_COUNTRY = aoi_country_lookup[INIT_AOI]
INIT_GEOHAZ = aoi_geohaz_lookup[INIT_AOI]
INIT_AVG_VEL_THR = 5.0
INIT_ADA_TYPE = "avgvel+"

INIT_ADA_CLASS_DIR = get_ada_location(
    ADA_CLASS_DIR, INIT_MODEL_DATE, INIT_EGMS_DATE,
    INIT_PRODUCT, INIT_COUNTRY, INIT_GEOHAZ, INIT_AOI,
    INIT_S1PATH, INIT_ADA_TYPE, INIT_AVG_VEL_THR
)
INIT_BOUNDS_DIR = get_aoi_bounds_location(
    AOI_BOUNDS_DIR, INIT_COUNTRY, INIT_GEOHAZ, INIT_AOI
)
PID_LOOKUP_DIR = get_pid_lookup_location(
    PID_FILE_LU_DIR, INIT_EGMS_DATE, INIT_PRODUCT,
    INIT_COUNTRY, INIT_GEOHAZ, INIT_AOI, INIT_S1PATH
)

egms_dates = os.listdir(
    re.sub("^(.*)/{egms_date}.*", "\\1", ADA_CLASS_DIR)
    .replace("{model_date}", INIT_MODEL_DATE)
)

ADAS_GDF = read_geo_data(
    os.path.join(INIT_ADA_CLASS_DIR, f"{INIT_AOI_CLEAN}_ada+_union.parquet")
).to_crs("EPSG:4326")
POINTS_GDF = read_geo_data(
    os.path.join(INIT_ADA_CLASS_DIR, f"{INIT_AOI_CLEAN}_points.parquet")
).to_crs("EPSG:4326")
POINTS_GDF.rename(
    columns={"class_label": "trend_class",
             "trend_subclass2": "trend_subclass",
             "label_prob": "mp_label_prob",
            },
    inplace=True
)

POINTS_GDF = POINTS_GDF.sjoin(ADAS_GDF).rename(columns={"index_right": "ada_id"})
ada_mean_velocity = POINTS_GDF.groupby("ada_id")["mean_velocity"].mean()
ADAS_GDF = ADAS_GDF.merge(ada_mean_velocity, left_index=True, right_index=True)
ADAS_GDF["mean_velocity_grp"] = create_velocity_groups(
    ADAS_GDF["mean_velocity"]
)
POINTS_GDF["mean_velocity_grp"] = create_velocity_groups(
    POINTS_GDF["mean_velocity"]
)

aoi_bounds_gdf = gpd.read_file(
    os.path.join(INIT_BOUNDS_DIR, INIT_AOI + ".geojson")
)
aoi_centre = aoi_bounds_gdf.centroid.to_crs("EPSG:4326").iloc[0]
aoi_centre_list = [aoi_centre.y, aoi_centre.x]
PID_LU_DF = pd.read_parquet(PID_LOOKUP_DIR)

ADA_COLOR_COLS = [
    "mean_velocity", "label_prob", "stable_prop",
    "ada_major_class", "ada_major_subclass"
]
POINT_COLOR_COLS = [
    "mean_velocity", "label_prob",
    "trend_class", "trend_subclass"
]


def get_trend_func(trend_type):
    """Return the model function to fit based on
    ML trend classification"""
    if trend_type in ["linear", "stable"]:
        trend_func = fa.linear_trend
    elif trend_type == "quadratic":
        trend_func = fa.quadratic_trend
    elif trend_type == "changepoint":
        trend_func = fa.linear_piecewise_trend
    elif trend_type == "step":
        trend_func = fa.step_trend
    else:
        raise NotImplementedError()
    return trend_func


# -----------------------------------------
# ---------------- LAYOUT -----------------
# -----------------------------------------

app = Dash(
    __name__,
    external_scripts=[chroma],
    prevent_initial_callbacks=True,
    external_stylesheets=[dbc.themes.YETI],
    suppress_callback_exceptions=True
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=LOGO, height="30px")),
                        dbc.Col(dbc.NavbarBrand("EGMS ADA+ viewer", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="https://placehold.co/",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            # dbc.Collapse(
            #     search_bar,
            #     id="navbar-collapse",
            #     is_open=False,
            #     navbar=True,
            # ),
        ]
    ),
    color="dark",
    dark=True,
)

# ------------- FILTER PANEL ----------------

ada_point_group = html.Div(
    [
        html.H6("View"),
        dbc.RadioItems(
            id="ada-point-radio",
            class_name="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "ADAs", "value": 1},
                {"label": "Points", "value": 2, "disabled": False},
            ],
            value=1,
        ),
    ],
    className="radio-group",
)
path_button_group = html.Div(
    [
        html.H6("Orbit"),
        dbc.RadioItems(
            id="path-radio",
            class_name="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Ascending", "value": "asc"},
                {"label": "Descending", "value": "dsc", "disabled": False},
            ],
            value="asc",
        ),
    ],
    className="radio-group",
)
egms_date_selection = html.Div(
    [
        html.H6("EGMS dataset"),
        render_dropdown(
            id="egms-dropdown",
            items={
                prettify_egms_date(egms_date): egms_date
                for egms_date in egms_dates
            },
        )
    ]
)
aoi_selection = html.Div(
    [
        html.H6("AOI selection"),
        render_dropdown(
            id="aoi-dropdown",
            items={val: key for key, val in aoi_name_lookup.items()}
        )
    ]
)
avgvel_selection = html.Div(
    [
        html.H6("Active velocity thr"),
        render_dropdown(
            id="avgvel-dropdown",
            items=["3.5mm", "5.0mm"],
            default="5.0mm"
        )
    ]
)
color_selection = html.Div(
    [
        html.H6("Colour by"),
        render_dropdown(
            id="color-dropdown",
            items=["mean_velocity", "ada_major_class", "ada_major_subclass"]
        )
    ]
)
get_data_button = html.Div(
    [
        html.H6(" "),
        dbc.Button(
            id="get-data-button",
            children="Load data",
            disabled=False
        )
    ]
)

empty_div = html.Div(id="empty-div", children=[""])

# ------------- TS Plots ----------------

ts_plot_tabs = dbc.Tabs(
    [
        dcc.Tab(
            dbc.Spinner(
                [
                    # html.P(
                    #     id="scatterplot-ts-header",
                    #     style={"font-weight": "bold"}
                    # ),
                    dash_table.DataTable(
                        id="trendfit-table",
                        style_data={
                            "whiteSpace": "normal",
                            "height": "auto",
                        },
                    ),
                    dcc.Graph(
                        id="scatterplot-ts",
                        figure=plot_blank_scatterplot()
                    )
                ]
            ),
            id="tab-tsplot",
            label="Trend fitting"
            ),
        dcc.Tab(
            dbc.Spinner(
                [
                    html.P(
                        id="resids-ts-header",
                        # style={"font-weight": "bold"},
                        children=[
                            html.Br(),
                            dbc.Tabs(
                                [
                                    dcc.Tab(
                                        dcc.Graph(
                                            id="resids-ts",
                                            figure=plot_blank_scatterplot("")
                                        ),
                                        id="resid-ts-plot",
                                        label="Time series"
                                    ),
                                    dcc.Tab(
                                        dcc.Graph(
                                            id="resids-fit",
                                            figure=plot_blank_scatterplot("")
                                        ),
                                        id="resid-fit-plot",
                                        label="Resids vs Fitted"
                                    ),
                                    dcc.Tab(
                                        dcc.Graph(
                                            id="resids-qq",
                                            figure=plot_blank_scatterplot("")
                                        ),
                                        id="resids-qq-plot",
                                        label="Q-Q"
                                    ),
                                ]
                            )
                        ]
                    ),                 
                ]
            ),
            id="tab-resids",
            label="Residuals",
            disabled=True
        ),
        dcc.Tab(
            dbc.Spinner(
                [
                    html.P(
                        id="season-ts-header",
                        children=[
                            html.Br(),
                            dbc.Tabs(
                                [
                                    dcc.Tab(
                                        [
                                            dash_table.DataTable(
                                                id="season-table",
                                                style_data={
                                                    "whiteSpace": "normal",
                                                    "height": "auto",
                                                },
                                            ),
                                            dcc.Graph(
                                                id="season-ts",
                                                figure=plot_blank_scatterplot("")
                                            )
                                        ],
                                        id="season-ts-plot",
                                        label="Time series"
                                    ),
                                    dcc.Tab(
                                        dcc.Graph(
                                            id="season-psd",
                                            figure=plot_blank_scatterplot("")
                                        ),
                                        id="season-psd-plot",
                                        label="PSD"
                                    ),
                                ]
                            )
                        ]
                    ),
                ]
            ),
            id="tab-season",
            label="Seasonality",
            disabled=True
        ),
        dcc.Tab(
            dbc.Spinner(
                [
                    html.P(
                        id="outliers-ts-header",
                        children=[
                            html.Br(),
                            dbc.Tabs(
                                [
                                    dcc.Tab(
                                        dcc.Graph(
                                            id="outliers-ts",
                                            figure=plot_blank_scatterplot("")
                                        ),
                                        id="outliers-ts-plot",
                                        label="Time series"
                                    ),
                                ]
                            )
                        ]
                    ),
                ]
            ),
            id="tab-outliers",
            label="Outliers",
            disabled=True
        ),
    ],
    id="tabs-tsplots"
)

filter_panel = html.Div(

    [
        # html.H6("Data selection filters"),
        dbc.Row(
            [
                dbc.Col(
                    egms_date_selection
                ),
                dbc.Col(
                    aoi_selection
                ),
                dbc.Col(
                    avgvel_selection
                ),
                dbc.Col(
                    dbc.Spinner(get_data_button, fullscreen=True)
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(ada_point_group),
                dbc.Col(path_button_group),
                dbc.Col(color_selection),
                dbc.Col(),
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(ts_plot_tabs),
            ]
        ),
    ]
)

# --------------- MAPS -------------------


def create_map_classes(data_col: str = "ada_major_class") -> list[str | int]:
    if data_col == "ada_major_class":
        classes = list(trend_class_cmap.keys())
    elif data_col == "ada_major_subclass":
        classes = list(trend_subclass_cmap.keys())
    elif data_col == "mean_velocity":
        classes = list(insar_vel_grp_cmap.keys())
    else:
        classes = None
    return classes


def create_map_colorscale(data_col: str = "ada_major_class"):
    if data_col == "ada_major_class":
        colorscale = list(trend_class_cmap.values())
    elif data_col == "ada_major_subclass":
        colorscale = list(trend_subclass_cmap.values())
    elif data_col == "mean_velocity":
        colorscale = list(insar_vel_grp_cmap.values())
    elif data_col == "mean_velocity_grp":
        colorscale = list(insar_vel_grp_cmap.values())
    elif data_col == "label_prob":
        colorscale = label_prob_colors
    elif data_col == "stable_prop":
        colorscale = stable_prop_colors
    else:
        colorscale = ["blue", "white", "red"]
    return colorscale


def create_map_cat_colorbar(data_col: str = "ada_major_class"):
    classes = create_map_classes(data_col)
    colorscale = create_map_colorscale(data_col)
    return dlx.categorical_colorbar(
        categories=classes,
        colorscale=colorscale,
        width=20, height=120,
        position="topright",
        id="map-colorbar"
    )


def create_hideout(data_col: str = "ada_major_class"):
    """Update the hideout parameter for maps dependant
    on the specified data metric"""
    if data_col == "mean_velocity":
        col_prop = "mean_velocity_grp"
    else:
        col_prop = data_col
    classes = create_map_classes(data_col)
    colorscale = create_map_colorscale(data_col)
    style = dict(
        weight=1,
        opacity=1,
        color="black",
        dashArray="3",
        fillOpacity=0.7,
    )
    return dict(
        classes=classes, colorscale=colorscale,
        style=style, colorProp=col_prop
    )


def create_points_hideout(data_col: str = "ada_major_class"):
    """Update the hideout parameter for maps dependant
    on the specified data metric"""
    # Check if continuous or discrete variable
    if data_col == "ada_major_class":
        col_prop = "trend_class"
    elif data_col == "ada_major_subclass":
        col_prop = "trend_subclass"
    elif data_col == "label_prob":
        col_prop = "mp_label_prob"
    elif data_col == "mean_velocity":
        col_prop = "mean_velocity_grp"
    else:
        col_prop = data_col
    out_dict = dict(
        colorProp=col_prop,
        circleOptions=dict(radius=2.5, stroke=False, fillOpacity=1),
        color_dict=metric_color_dict[col_prop],
    )
    return out_dict


def create_poly_geojson(data, data_col, style_handle, id):
    return dl.GeoJSON(
        data=data.__geo_interface__,
        style=style_handle,
        onEachFeature=on_each_poly_feature,
        zoomToBounds=False,
        zoomToBoundsOnClick=True,
        hoverStyle=arrow_function(
            dict(weight=4, color="#222222", dashArray="")
        ),
        hideout=create_hideout(data_col),
        id=id,
    )


def create_points_geojson(data, data_col, style_handle, id):
    return dl.GeoJSON(
        data=data.__geo_interface__,
        interactive=True,
        pointToLayer=style_handle,
        onEachFeature=on_each_feature,
        zoomToBounds=False,
        hideout=create_points_hideout(data_col),
        id=id,
    )


poly_style_handle = assign(
    """function(feature, context){
    const {classes, colorscale, style, colorProp} = context.hideout;
    const value = feature.properties[colorProp];
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            style.fillColor = colorscale[i];
        }
    }
    return style;}"""
)
cat_points_style_handle = assign(
    """function(feature, latlng, context){
    const {colorProp, circleOptions, color_dict} = context.hideout;
    const value  = feature.properties[colorProp];
    circleOptions.fillColor = color_dict[value];
    return L.circleMarker(latlng, circleOptions);}"""
)
on_each_feature = assign(
    """function(feature, layer, context){
    layer.bindTooltip(
        `PID: ${feature.properties.pid}<br>
        Mean velocity: ${feature.properties.mean_velocity.toFixed(2)}mm/yr<br>
        Label prob: ${(feature.properties.mp_label_prob).toFixed(2)}<br>
        Trend class: ${feature.properties.trend_class}<br>
        Trend subclass: ${feature.properties.trend_subclass}<br>`)
    }"""
)
on_each_poly_feature = assign(
    """function(feature, layer, context){
    layer.bindTooltip(
        `N active MPs: ${feature.properties.n_ada_points}<br>
        Mean velocity: ${feature.properties.mean_velocity.toFixed(2)}mm/yr<br>
        Stable prop: ${(feature.properties.stable_prop*100).toFixed(2)}%<br>
        Avg. label prob: ${(feature.properties.label_prob).toFixed(2)}<br>
        ADA major class: ${feature.properties.ada_major_class}<br>
        ADA major subclass: ${feature.properties.ada_major_subclass}<br>`)
    }"""
)
# insar_vel_points_style_handle = assign(
#     """function(feature, latlng, context){
#     const {min, max, colorscale, circleOptions, colorProp} = context.hideout;
#     const csc = chroma.scale(colorscale).domain([min, max]);
#     circleOptions.fillColor = csc(feature.properties[colorProp]);
#     return L.circleMarker(latlng, circleOptions);}"""
# )
# insar_vel_poly_style_handle = assign(
#     """function(feature, context){
#     const {min, max, colorscale, style, colorProp} = context.hideout;
#     const csc = chroma.scale(colorscale).domain([min, max]);
#     style.fillColor = csc(feature.properties[colorProp]);
#     return style;}"""
# )


poly_map_geojson = create_poly_geojson(
    ADAS_GDF, "ada_major_class", poly_style_handle,
    "map-polyg-geojson"
)
points_map_geojson = create_points_geojson(
    POINTS_GDF, "ada_major_class", cat_points_style_handle,
    "map-points-geojson"
)
cat_colorbar = create_map_cat_colorbar("mean_velocity")

tile_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
tile_attribution = "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"

leaflet_map = dl.Map(
    [
        dl.TileLayer(url=tile_url, attribution=tile_attribution),
        poly_map_geojson,
        cat_colorbar,
        dl.ScaleControl(
            position="bottomleft",
            imperial=False
        )
    ],
    center=aoi_centre_list,
    zoom=10,
    style={
        "height": "90vh",
        "width": "100%"
    },
    id="leaflet-map")


def serve_layout():
    session_id = str(uuid.uuid4())
    return dbc.Container(
        children=[
            dcc.Store(id="session-id", data=session_id),
            dcc.Store(id="poly-map-features", data=[], storage_type="session"),
            dcc.Store(id="point-map-features", data=[], storage_type="session"),
            dcc.Store(id="egms-ts", data=[], storage_type="session"),
            dcc.Store(id="fcf-output", data=[], storage_type="session"),
            dcc.Store(id="seasonality", data=[], storage_type="session"),
            dcc.Store(id="psd", data=[], storage_type="session"),
            dcc.Store(id="freqs", data=[], storage_type="session"),
            dcc.Store(id="label-probs", data=[], storage_type="session"),
            navbar,
            html.Div(style={"height": "2px"}),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Spinner(leaflet_map),
                        width=7),
                    dbc.Col(
                        filter_panel,
                        width=5
                    )
                ]
            )
        ],
        style={"maxWidth": "1920px"},
    )


app.layout = serve_layout


# ---------------------------------------------------------
# --------------------CALLBACKS----------------------------
# ---------------------------------------------------------


# ---------------- Filters/components ---------------------

# add callback for toggling the collapse on small screens
@callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(
    Output("aoi-dropdown", "options"),
    Input("egms-dropdown", "value"),
)
def update_aoi_dropdown(egms_date):
    ada_dir = re.sub(r"^(.*)/{country}.*", "\\1", ADA_CLASS_DIR)
    ada_dir = ada_dir.replace("{egms_date}", egms_date)
    ada_dir = ada_dir.replace("{model_date}", INIT_MODEL_DATE)
    ada_dir = ada_dir.replace("{product}", INIT_PRODUCT)
    countries = os.listdir(ada_dir)
    aoi_list = []
    for country in countries:
        ghaz_dir = os.path.join(ada_dir, country, "adas")
        for ghaz in os.listdir(ghaz_dir):
            aoi_list.append(
                os.listdir(os.path.join(ghaz_dir, ghaz))
            )
    aoi_list = [x for y in aoi_list for x in y]
    out_list = [
        {"label": aoi_name_lookup[aoi], "value": aoi}
        for aoi in aoi_list if aoi in aoi_name_lookup.keys()
    ]
    return out_list


@callback(
    Output("avgvel-dropdown", "options"),
    [Input("egms-dropdown", "value"),
     Input("aoi-dropdown", "value"),
     ]
)
def update_avgvel_dropdown(egms_date, aoi_name):
    ada_dir = re.sub(r"^(.*)/{avg_vel_thr}.*", "\\1", ADA_CLASS_DIR)
    ada_dir = ada_dir.replace("{egms_date}", egms_date)
    ada_dir = ada_dir.replace("{model_date}", INIT_MODEL_DATE)
    ada_dir = ada_dir.replace("{product}", INIT_PRODUCT)
    ada_dir = ada_dir.replace("{country}", aoi_country_lookup[aoi_name])
    ada_dir = ada_dir.replace("{geohaz_type}", aoi_geohaz_lookup[aoi_name])
    ada_dir = ada_dir.replace("{aoi_name}", aoi_name)
    ada_dir = ada_dir.replace("{s1_path}", INIT_S1PATH)
    ada_dir = ada_dir.replace("{ada_type}", INIT_ADA_TYPE)

    avg_vels = os.listdir(ada_dir)
    out_list = [
        {"label": avgvel, "value": avgvel[:-2]}
        for avgvel in avg_vels
    ]
    return out_list


@callback(
        Output("ada-point-radio", "value"),
        Output("leaflet-map", "center"),
        Output("leaflet-map", "zoom"),
        [
            Input("get-data-button", "n_clicks")
        ],
        [
            State("egms-dropdown", "value"),
            State("aoi-dropdown", "value"),
            State("avgvel-dropdown", "value"),
        ],
        prevent_initial_call=True
)
def update_aoi_data(n_clicks, egmsdate, aoiname, avgvel):
    """Update the main mapping dataframes"""
    if n_clicks > 0:

        global ADAS_GDF
        global POINTS_GDF
        global PID_LU_DF
        global PID_LOOKUP_DIR

        ada_class_dir = get_ada_location(
            ADA_CLASS_DIR, INIT_MODEL_DATE, egmsdate,
            INIT_PRODUCT, aoi_country_lookup[aoiname],
            aoi_geohaz_lookup[aoiname], aoiname,
            INIT_S1PATH, INIT_ADA_TYPE, avgvel
        )
        bounds_dir = get_aoi_bounds_location(
            AOI_BOUNDS_DIR, aoi_country_lookup[aoiname],
            aoi_geohaz_lookup[aoiname], aoiname
        )
        PID_LOOKUP_DIR = get_pid_lookup_location(
            PID_FILE_LU_DIR, egmsdate, INIT_PRODUCT,
            aoi_country_lookup[aoiname], aoi_geohaz_lookup[aoiname],
            aoiname, INIT_S1PATH
        )

        aoi_clean = re.sub(r"^(.*)_\d+$", "\\1", aoiname)

        ADAS_GDF = read_geo_data(
            os.path.join(ada_class_dir, f"{aoi_clean}_ada+_union.parquet")
        ).to_crs("EPSG:4326")
        POINTS_GDF = read_geo_data(
            os.path.join(ada_class_dir, f"{aoi_clean}_points.parquet")
        ).to_crs("EPSG:4326")
        POINTS_GDF.rename(
            columns={
                "class_label": "trend_class",
                "trend_subclass2": "trend_subclass",
                "label_prob": "mp_label_prob",
            },
            inplace=True
        )
        POINTS_GDF = POINTS_GDF.sjoin(ADAS_GDF).rename(columns={"index_right": "ada_id"})
        ada_mean_velocity = POINTS_GDF.groupby("ada_id")["mean_velocity"].mean()
        ADAS_GDF = ADAS_GDF.merge(
            ada_mean_velocity, left_index=True, right_index=True    
        )
        ADAS_GDF["mean_velocity_grp"] = create_velocity_groups(
            ADAS_GDF["mean_velocity"]
        )
        POINTS_GDF["mean_velocity_grp"] = create_velocity_groups(
            POINTS_GDF["mean_velocity"]
        )
        PID_LU_DF = pd.read_parquet(PID_LOOKUP_DIR)

        aoi_bounds_gdf = gpd.read_file(
            os.path.join(bounds_dir, aoiname + ".geojson")
        )
        aoi_centre = aoi_bounds_gdf.centroid.to_crs("EPSG:4326").iloc[0]
        aoi_centre_list = [aoi_centre.y, aoi_centre.x]

        print(POINTS_GDF.head())
        print(POINTS_GDF.crs)
        print(aoi_centre_list)
        return 1, aoi_centre_list, 10


# ---------------- MAPS ---------------------

@callback(
    Output("leaflet-map", "children"),
    Output("scatterplot-ts", "figure", allow_duplicate=True),
    [Input("ada-point-radio", "value")],
    State("color-dropdown", "value"),
    prevent_initial_call=True
)
def update_poly_points_map(ada_point_val, color_col):
    if ada_point_val == 1:
        geojson = create_poly_geojson(
            ADAS_GDF, color_col, poly_style_handle,
            "map-polyg-geojson"
        )
    if ada_point_val == 2:
        geojson = create_points_geojson(
            POINTS_GDF, color_col, cat_points_style_handle,
            "map-points-geojson"
        )

    children = [
        dl.TileLayer(url=tile_url, attribution=tile_attribution),
        geojson,
        create_map_cat_colorbar(color_col),
        dl.ScaleControl(
            position="bottomleft",
            imperial=False
        )
    ]
    return children, plot_blank_scatterplot()


@callback(
    Output("map-polyg-geojson", "hideout"),
    [Input("color-dropdown", "value")],
    State("ada-point-radio", "value"),
    prevent_initial_call=True
)
def update_poly_map_colour(color_col, ada_point_val):
    if ada_point_val == 2:
        raise PreventUpdate
    hideout = create_hideout(color_col)
    classes = create_map_classes(color_col)
    colorscale = create_map_colorscale(color_col)
    return hideout


@callback(
    Output("map-colorbar", "categories"),
    Output("map-colorbar", "colorscale"),
    [Input("color-dropdown", "value")],
    prevent_initial_call=True
)
def update_map_colourbar(color_col):
    classes = create_map_classes(color_col)
    colorscale = create_map_colorscale(color_col)
    return classes, colorscale


@callback(
    Output("map-points-geojson", "hideout"),
    [Input("color-dropdown", "value")],
    State("ada-point-radio", "value"),
    prevent_initial_call=True
)
def update_points_map_colour(color_col, ada_point_val):
    if ada_point_val == 1:
        raise PreventUpdate
    hideout = create_points_hideout(color_col)
    return hideout


@callback(
    Output("poly-map-features", "data"),
    [Input("map-polyg-geojson", "clickData")],
    prevent_initial_call=True
)
def update_poly_plots(click_data):
    if click_data is None:
        raise PreventUpdate
    return click_data["properties"]


@callback(
    Output("point-map-features", "data"),
    Output("egms-ts", "data", allow_duplicate=True),
    [Input("map-points-geojson", "clickData")],
    prevent_initial_call=True
)
def get_ts_point_data(click_data):
    if click_data is None:
        raise PreventUpdate
    pid = click_data["properties"]["pid"]
    ts_filename = get_ts_filename(
        pid, PID_LU_DF,
        PID_LOOKUP_DIR.replace("lookups/", "")
    )
    pid_df = read_single_ts(ts_filename, pid)
    dates = get_date_cols(pid_df)
    ts_df = pid_df[dates]
    ts_df = resample_egms_data(ts_df[dates], dates)
    return click_data["properties"], ts_df.to_json()


# ---------------- CHARTS ---------------------

@callback(
    Output("fcf-output", "data", allow_duplicate=True),
    [
        Input("session-id", "data"),
        Input("point-map-features", "data"),
        Input("egms-ts", "data"),
     ],
    # [State("nsteps-input", "value"),
    #  State("season-freq-slider", "value"),
    #  State("psd-input", "value"),],
    prevent_initial_call=True
)
def classify_timeseries(session_id, point_features, egms_ts):
    nsteps, period, psd_thr = (3, [0.5, 2.0], 30.0)  # need to add filters
    trend_type = point_features["trend_class"]
    sample_ts = convert_json_to_dataframe(egms_ts)
    dates = sample_ts.columns
    sr = get_most_common_sr(dates)
    annual_timesteps = 365.25/sr
    insar_cls = incls.InsarTSClassifier(
            first_date=dates[0],
            sample_rate=1/annual_timesteps,
            trends=[get_trend_func(trend_type)],
            init_trend=trend_type,
            max_n_breaks=3 if nsteps is None else nsteps,
            min_season_freq=period[0],
            max_season_freq=period[1],
            psd_thr=30 if psd_thr is None else psd_thr
        )
    fcf_output = insar_cls.fit(
        sample_ts.iloc[0].astype(float).interpolate(method="linear")
    )
    # Convert complex numbers to real and numpy array to list
    # for json serialization
    out_dict = {}
    for k, v in vars(fcf_output).items():
        if isinstance(v, np.ndarray):
            out_dict[k] = np.real(v).tolist()
        if isinstance(v, str | list | tuple | float | int) and k != "trends":
            out_dict[k] = v
    return out_dict


@callback(
    Output("scatterplot-ts", "figure", allow_duplicate=True),
    [
        Input("fcf-output", "data")
    ],
    [
        State("point-map-features", "data"),
        State("egms-ts", "data")
    ],
    prevent_initial_call=True
)
def update_scatterplot_ts(fcf_output, point_features, egms_ts):
    if egms_ts is not None:
        egms_ts = pd.DataFrame.from_dict(json.loads(egms_ts))
        fcf_output = json.loads(json.dumps(fcf_output))

        fig = plot_time_series_decomp(
            egms_ts.iloc[0],
            fcf_output["trend_vals"],
            fcf_output["ffilt"],
            np.array(
                [datetime.strptime(d, "%Y%m%d") for d in egms_ts.columns]
            )
        )
        return fig
    else:
        return plot_blank_scatterplot()


# @callback(
#     Output("trendfit-table", "data"),
#     Output("trendfit-table", "columns"),
#     [Input("egms-ts", "data"),
#      Input("fcf-output", "data"),
#      Input("label-probs", "data")],
#     [State("trend-dropdown", "value"),
#      State("pid-dropdown", "value")],
#     prevent_initial_call=True
# )
# def update_trndfit_table(egms_ts, fcf_output, lab_probs, trend_type, pid):
#     if egms_ts is not None:
#         fcf_output = json.loads(json.dumps(fcf_output))
#         rmse = fcf_output["rmse"]
#         seg_rmse = fcf_output.get("seg_rmse", np.nan)
#         columns = ["PID", "Classified trend", "Class probability", "RMSE", "Max Seg. RMSE"]
#         df = pd.DataFrame(
#             data={
#                 "PID": [pid],
#                 "Classified trend": [trend_type.capitalize()],
#                 "Class probability": [round(lab_probs, 3)],
#                 "RMSE": [round(rmse, 3)],
#                 "Max Seg. RMSE": [round(np.nanmax(seg_rmse), 3)]
#             },
#             columns=columns
#         )
#         return (df.to_dict("records"),
#                 [{"name": c, "id": c} for c in df.columns])
#     return [], []


# @callback(
#     Output("pid-dropdown", "options", allow_duplicate=True),
#     Input("trend-dropdown", "value"),
#     prevent_initial_call=True
# )
# def update_pid_list(trend_type):
#     return cvd.get_pids_to_process(egms_ts_df, trend_type)


# @callback(
#     Output("pid-dropdown", "options", allow_duplicate=True),
#     Input("output-tsmetrics-button", "n_clicks"),
#     State("trend-dropdown", "value"),
#     State("newtrend-dropdown", "value"),
#     State("cp-date-input", "value"),
#     State("pid-dropdown", "value"),
#     prevent_initial_call=True
# )
# def output_manual_checks(
#         n_clicks, trend_class, newtrend_class, date_input, pid):
#     if n_clicks is None:
#         raise dash.exceptions.PreventUpdate()
#     else:
#         out_df = egms_ts_df[egms_ts_df["pid"] == pid][["pid", "filename"]]
#         out_df["ml_class"] = trend_class
#         out_df["manual_class"] = newtrend_class
#         out_df["cp_locations"] = date_input
#         output_manual_check_to_disk(out_df)
#         return cvd.get_pids_to_process(egms_ts_df, trend_class, pid)


# @callback(
#     Output("resids-ts", "figure"),
#     [Input("egms-ts", "data"),
#      Input("fcf-output", "data")],
#     [State("trend-dropdown", "value"),
#      State("pid-dropdown", "value")],
#     prevent_initial_call=True
# )
# def update_resids_plot(egms_ts, fcf_output, trend_type, pid):
#     if egms_ts is not None:
#         egms_ts = pd.DataFrame.from_dict(json.loads(egms_ts))
#         fcf_output = json.loads(json.dumps(fcf_output))
#         fig = plot_time_series_residuals(
#             (
#                 egms_ts.iloc[0]
#                 - np.array(fcf_output["trend_vals"])
#                 - np.array(fcf_output["ffilt"])
#             ),
#             np.array(
#                 [datetime.strptime(d, "%Y%m%d") for d in egms_ts.columns]
#             )
#         )
#         return fig
#     return plot_blank_scatterplot()


# @callback(
#     Output("resids-fit", "figure"),
#     [Input("egms-ts", "data"),
#      Input("fcf-output", "data")],
#     [State("trend-dropdown", "value"),
#      State("pid-dropdown", "value")],
#     prevent_initial_call=True
# )
# def update_resids_fit_plot(egms_ts, fcf_output, trend_type, pid):
#     if egms_ts is not None:
#         egms_ts = pd.DataFrame.from_dict(json.loads(egms_ts))
#         fcf_output = json.loads(json.dumps(fcf_output))
#         fig = plot_fitted_residuals(
#             (
#                 np.array(fcf_output["trend_vals"])
#                 + np.array(fcf_output["ffilt"])
#             ),
#             (
#                 egms_ts.iloc[0]
#                 - np.array(fcf_output["trend_vals"])
#                 - np.array(fcf_output["ffilt"])
#             ),
#         )
#         return fig
#     return plot_blank_scatterplot()


# @callback(
#     Output("resids-qq", "figure"),
#     [Input("egms-ts", "data"),
#      Input("fcf-output", "data")],
#     [State("trend-dropdown", "value"),
#      State("pid-dropdown", "value")],
#     prevent_initial_call=True
# )
# def update_resids_qq_plot(egms_ts, fcf_output, trend_type, pid):
#     if egms_ts is not None:
#         egms_ts = pd.DataFrame.from_dict(json.loads(egms_ts))
#         fcf_output = json.loads(json.dumps(fcf_output))
#         fig = plot_qq(
#             (
#                 egms_ts.iloc[0]
#                 - np.array(fcf_output["trend_vals"])
#                 - np.array(fcf_output["ffilt"])
#             ),
#         )
#         return fig
#     return plot_blank_scatterplot()


# @callback(
#     Output("season-psd", "figure"),
#     [Input("egms-ts", "data"),
#      Input("fcf-output", "data")],
#     [State("season-freq-slider", "value"),
#      State("psd-input", "value")],
#     prevent_initial_call=True
# )
# def update_psd_plot(egms_ts, fcf_output, freq_thr, psd_thr):
#     if egms_ts is not None:
#         egms_ts = pd.DataFrame.from_dict(json.loads(egms_ts))
#         fcf_output = json.loads(json.dumps(fcf_output))
#         fig = plot_psd(
#             np.array(fcf_output["psd"]),
#             np.array(fcf_output["freq"]),
#             fcf_output["psd_thr"],
#             [fcf_output["min_season_freq"], fcf_output["max_season_freq"]],
#         )
#         return fig
#     return plot_blank_scatterplot()


# @callback(
#     Output("season-ts", "figure"),
#     [Input("egms-ts", "data"),
#      Input("fcf-output", "data")],
#     prevent_initial_call=True
# )
# def update_season_ts_plot(egms_ts, fcf_output):
#     if egms_ts is not None:
#         egms_ts = pd.DataFrame.from_dict(json.loads(egms_ts))
#         fcf_output = json.loads(json.dumps(fcf_output))
#         if np.sum(np.abs(fcf_output["ffilt"])) > 0:
#             peaks, troughs = fcf_output["season_peaks"][0], fcf_output["season_peaks"][2]
#         else:
#             peaks, troughs = [], []
#         fig = plot_seasonality_ts(
#             np.array(fcf_output["ffilt"]),
#             np.array(
#                 [datetime.strptime(d, "%Y%m%d") for d in egms_ts.columns]
#             ),
#             peaks,
#             troughs,
#         )
#         return fig
#     return plot_blank_scatterplot()


# @callback(
#     Output("season-table", "data"),
#     Output("season-table", "columns"),
#     [Input("egms-ts", "data"),
#      Input("fcf-output", "data")],
#     prevent_initial_call=True
# )
# def update_season_table(egms_ts, fcf_output):
#     if egms_ts is not None:
#         egms_ts = pd.DataFrame.from_dict(json.loads(egms_ts))
#         fcf_output = json.loads(json.dumps(fcf_output))
#         if np.sum(np.abs(fcf_output["ffilt"])) > 0:
#             peak_amp = fcf_output["season_pkpk_amp"]
#             rms = fcf_output["season_rms"]
#             dates = [
#                 datetime.strptime(d, "%Y%m%d") for d in
#                 egms_ts.columns[fcf_output["season_peaks"][0]]
#             ]
#             period = [d.days for d in np.diff(dates)]
#             columns = ["", "Period (days)", "Peak-peak amp (mm)", "RMS"]
#             df = pd.DataFrame(
#                 data={
#                     "": ["Mean", "Sdev"],
#                     "Period (days)": [round(np.mean(period), 2),
#                                       round(np.std(period), 2)],
#                     "Peak-peak amp (mm)": [round(np.mean(np.abs(peak_amp)), 2),
#                                            round(np.std(np.abs(peak_amp)), 2)],
#                     "RMS": [round(rms, 2), ""]
#                 },
#                 columns=columns
#             )
#             return (df.to_dict("records"),
#                     [{"name": c, "id": c} for c in df.columns])
#         return [], []
#     return [], []


# @callback(
#     Output("outliers-ts", "figure"),
#     [Input("egms-ts", "data")],
#     [State("trend-outlier-thr", "value")],
#     prevent_initial_call=True
# )
# def update_outliers_ts_plot(egms_ts, outlier_thr):
#     if egms_ts is not None:
#         egms_ts = pd.DataFrame.from_dict(json.loads(egms_ts))
#         egms_ts = egms_ts.dropna(axis=1)
#         fig = plot_ts_outliers(
#             egms_ts.iloc[0],
#             np.array(
#                 [datetime.strptime(d, "%Y%m%d") for d in egms_ts.columns]
#             ),
#             thr=8.45 if outlier_thr is None else outlier_thr,
#         )
#         return fig
#     return plot_blank_scatterplot()


@callback(
    Output("tab-resids", "disabled", allow_duplicate=True),
    Output("tab-season", "disabled", allow_duplicate=True),
    Output("tab-outliers", "disabled", allow_duplicate=True),
    Input("egms-ts", "data"),
    prevent_initial_call=True
)
def disable_decomp_tabs(egms_ts):
    if egms_ts is not None:
        return False, False, False
    return True, True, True


# @callback(
#     Output("trend-collapse", "is_open"),
#     [Input("trend-collapse-button", "n_clicks")],
#     [State("trend-collapse", "is_open")],
# )
# def trend_toggle_collapse(n, is_open):
#     if n:
#         return not is_open
#     return is_open


# @callback(
#     Output("season-collapse", "is_open"),
#     [Input("season-collapse-button", "n_clicks")],
#     [State("season-collapse", "is_open")],
# )
# def season_toggle_collapse(n, is_open):
#     if n:
#         return not is_open
#     return is_open


if __name__ == '__main__':
    app.run(debug=True)

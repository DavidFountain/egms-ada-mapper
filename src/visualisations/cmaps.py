# -------------------------
# -------- CMAPs ----------
# -------------------------

adatype_cmap = {
    "avgvel+": "#01FDF6",
    "avgvel": "#24272B",
    "both": "#0D00A4"
}

adasubtype_cmap = {
    "Separate": "#C62E65",
    "Extended": "#2F1847"
}

# Trend type
trend_class_cmap = {
    "stable": "#4daf4a",
    "linear": "#377eb8",
    "quadratic": "#ff7f00",
    "changepoint": "#984ea3",
    "step": "#e41a1c"
}

# Trend subtype
trend_subclass_cmap = {
    "stable": "green",
    "active-stable": "blue",
    "gradual-deceleration": "lightblue",
    "stable-active": "red",
    "gradual-acceleration": "salmon",
    "rebound": "deeppink",
    "active-constant": "orange",
    "active-acceleration": "purple",
    "active-deceleration": "yellow",
    "active-dir-change": "black",
}

# InSAR velocity
insar_vel_grp_cmap = {
    "<-10": "red",
    "<-6": "orange",
    "<-2": "yellow",
    "[-2, 2]": "lime",
    ">2": "aquamarine",
    ">6": "darkturquoise",
    ">10": "blue",
    }

# InSAR velocity
insar_velocity_colors = [
    "darkred", "red", "orange", "yellow", "lime",
    "aquamarine", "darkturquoise", "blue", "darkblue"
]

# Label probabilities
label_prob_colors = ["blue", "white"]

# Stable proportion
stable_prop_colors = ["red", "white"]

# For hideout
metric_color_dict = {
    "ada_major_class": trend_class_cmap,
    "ada_major_subclass": trend_subclass_cmap,
    "trend_class": trend_class_cmap,
    "trend_subclass": trend_subclass_cmap,
    "mean_velocity_grp": insar_vel_grp_cmap,
    "mean_velocity": insar_velocity_colors,
    "label_prob": label_prob_colors,
    "mp_label_prob": label_prob_colors,
    "stable_prop": stable_prop_colors,
}

window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, context) {
            const {
                classes,
                colorscale,
                style,
                colorProp
            } = context.hideout;
            const value = feature.properties[colorProp];
            for (let i = 0; i < classes.length; ++i) {
                if (value == classes[i]) {
                    style.fillColor = colorscale[i];
                }
            }
            return style;
        },
        function1: function(feature, latlng, context) {
            const {
                colorProp,
                circleOptions,
                color_dict
            } = context.hideout;
            const value = feature.properties[colorProp];
            circleOptions.fillColor = color_dict[value];
            return L.circleMarker(latlng, circleOptions);
        },
        function2: function(feature, layer, context) {
            layer.bindTooltip(
                `PID: ${feature.properties.pid}<br>
                Mean velocity: ${feature.properties.mean_velocity.toFixed(2)}mm/yr<br>
                Label prob: ${(feature.properties.mp_label_prob).toFixed(2)}<br>
                Trend class: ${feature.properties.trend_class}<br>
                Trend subclass: ${feature.properties.trend_subclass}<br>`)
        },
        function3: function(feature, layer, context) {
            layer.bindTooltip(
                `N active MPs: ${feature.properties.n_ada_points}<br>
                Mean velocity: ${feature.properties.mean_velocity.toFixed(2)}mm/yr<br>
                Stable prop: ${(feature.properties.stable_prop*100).toFixed(2)}%<br>
                Avg. label prob: ${(feature.properties.label_prob).toFixed(2)}<br>
                ADA major class: ${feature.properties.ada_major_class}<br>
                ADA major subclass: ${feature.properties.ada_major_subclass}<br>`
            )
        },
        function4: function(feature, latlng, context) {
            const {
                min,
                max,
                colorscale,
                circleOptions,
                colorProp
            } = context.hideout;
            const csc = chroma.scale(colorscale).domain([min, max]);
            circleOptions.fillColor = csc(feature.properties[colorProp]);
            return L.circleMarker(latlng, circleOptions);
        },
        function5: function(feature, context) {
            const {
                min,
                max,
                colorscale,
                style,
                colorProp
            } = context.hideout;
            const csc = chroma.scale(colorscale).domain([min, max]);
            style.fillColor = csc(feature.properties[colorProp]);
            return style;
        }
    }
});
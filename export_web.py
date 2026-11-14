# -*- coding: utf-8 -*-
"""
export_web.py: turn the layers_3D Plotly figure into a phone-friendly index.html
for GitHub Pages (https://jasonbott124.github.io/takahe-3d/).

Usage (in the session where you already build the figure):
    import layers_3D
    from export_web import CURRENT_AGES, export_for_web
    layers_3D.AGE_KEY.update(CURRENT_AGES)      # BEFORE plot_layers_3d
    fig = layers_3D.plot_layers_3d(..., step=10, res=2000)   # lighter for phones
    export_for_web(fig, "index.html")
Then copy index.html into the takahe-3d repo and push.
"""
import os
import layers_3D

# WD2014 tie at Y67a (Buizert 2015; Sigl 2016), in years.
# D3 / D5 are not in the current dating list: check before publishing.
CURRENT_AGES = {"D1": 592.0, "D2": 1185.0, "D7": 1778.0, "D4": 2224.0, "D6": 4260.0}

BG = "#0b1424"
FG = "#e8edf3"


def export_for_web(fig, out="index.html", title="Mt. Takahe: bed, ice surface and dated isochrones"):
    # labels + coupled surface/contour opacity slider (safe to re-run on any version of the figure)
    layers_3D.style_labels(fig)
    layers_3D.add_surface_opacity_slider(fig)

    axis = dict(color=FG, gridcolor="rgba(255,255,255,0.15)", backgroundcolor=BG,
                showbackground=True, zerolinecolor="rgba(255,255,255,0.25)",
                title_font=dict(size=13, family=layers_3D.LABEL_FONT))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15, color=FG, family=layers_3D.LABEL_FONT)),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=FG, size=12),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(11,20,36,0.6)", font=dict(size=11, color=FG),
                    itemclick="toggle", itemdoubleclick="toggleothers"),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis),
    )
    if fig.layout.sliders:
        fig.layout.sliders[0].update(font=dict(color=FG), bgcolor="#2a3a52",
                                     activebgcolor="#f7d417", bordercolor=FG,
                                     currentvalue=dict(font=dict(color=FG)))
    fig.update_traces(selector=dict(type="surface"),
                      colorbar=dict(tickfont=dict(color=FG), title_font=dict(color=FG), thickness=12))
    fig.write_html(
        out, include_plotlyjs="cdn", full_html=True,
        default_width="100%", default_height="100vh",
        config=dict(responsive=True, displaylogo=False, scrollZoom=True,
                    modeBarButtonsToRemove=["toImage", "resetCameraLastSave3d"]),
    )
    # phone viewport + no page margin
    with open(out, encoding="utf-8") as f:
        html = f.read()
    html = html.replace(
        "<head>",
        '<head><meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Takahe 3D</title><style>html,body{{margin:0;background:{BG}}}</style>", 1)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    mb = os.path.getsize(out) / 1e6
    print(f"wrote {out}: {mb:.1f} MB" + ("  <- heavy for phones; raise step/res" if mb > 15 else ""))

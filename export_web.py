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
    # sliders: surface/contour opacity + live vertical exaggeration (title follows the slider)
    meta = dict(fig.layout.meta or {})
    if meta:
        meta["title_base"] = title
        fig.update_layout(meta=meta)
    layers_3D.add_surface_opacity_slider(fig)
    layers_3D.add_vertical_exaggeration_slider(fig)
    ve = meta.get("vert_exag")
    full_title = f"{title} ({ve:g}\u00d7 vertical exaggeration)" if ve else title

    axis = dict(color=FG, gridcolor="rgba(255,255,255,0.15)", backgroundcolor=BG,
                showbackground=True, zerolinecolor="rgba(255,255,255,0.25)",
                title_font=dict(size=13, family=layers_3D.LABEL_FONT))
    fig.update_layout(
        title=dict(text=full_title, x=0.5, font=dict(size=15, color=FG, family=layers_3D.LABEL_FONT)),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=FG, size=12),
        margin=dict(l=0, r=0, t=40, b=0),
        # light legend box (as in the original) so the age-coloured layer lines stay visible
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)", font=dict(size=11, color="#13243a"),
                    itemclick="toggle", itemdoubleclick="toggleothers"),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis),
    )
    for sl in (fig.layout.sliders or []):
        sl.update(font=dict(color=FG), bgcolor="#2a3a52", activebgcolor="#f7d417", bordercolor=FG,
                  currentvalue=dict(font=dict(color=FG)))
    fig.update_traces(selector=dict(type="surface"),
                      colorbar=dict(tickfont=dict(color=FG), title_font=dict(color=FG), thickness=12))
    # never a fixed pixel size (plot_Surface_3D sets 1920x1080): fill the container instead
    fig.layout.width = None
    fig.layout.height = None
    fig.update_layout(autosize=True)

    import plotly.io as pio
    plot_div = pio.to_html(
        fig, include_plotlyjs="cdn", full_html=False, div_id="plot",
        default_width="100%", default_height="100%",
        config=dict(responsive=True, displaylogo=False, scrollZoom=True,
                    modeBarButtonsToRemove=["toImage", "resetCameraLastSave3d"]),
    )
    html = PAGE.replace("{BG}", BG).replace("{FG}", FG).replace("{PLOT}", plot_div)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    mb = os.path.getsize(out) / 1e6
    print(f"wrote {out}: {mb:.1f} MB" + ("  <- heavy for phones; raise step/res" if mb > 15 else ""))


# One-screen page: figure fills the space, native sliders in a compact bar underneath
# (Plotly's own sliders are hard to grab on a phone because touches rotate the scene).
PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Takahe 3D</title>
<style>
  html,body{margin:0;height:100%;overflow:hidden;background:{BG};color:{FG};
            font:12px/1.3 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  #wrap{display:flex;flex-direction:column;height:100vh;height:100dvh}
  #plotbox{flex:1;min-height:0}
  #plotbox > div{height:100%}
  #ctl{display:flex;flex-wrap:wrap;align-items:center;gap:4px 14px;padding:4px 10px 8px;
       border-top:1px solid rgba(255,255,255,.12)}
  .row{flex:1 1 200px;min-width:0}
  .row span{display:block;opacity:.85}
  input[type=range]{width:100%;margin:2px 0;height:26px;accent-color:#f7d417}
  @media (max-width:700px){ .modebar-container{display:none} }
  button{background:#2a3a52;color:{FG};border:1px solid rgba(255,255,255,.3);border-radius:4px;
         padding:6px 10px;font:inherit}
</style>
</head>
<body>
<div id="wrap">
  <div id="plotbox">{PLOT}</div>
  <div id="ctl"><button id="leg" type="button">Legend</button></div>
</div>
<script>
(function init(){
  var gd = document.getElementById('plot');
  if (!gd || !gd.layout || !window.Plotly) { return setTimeout(init, 100); }
  var ctl = document.getElementById('ctl');
  var small = window.innerWidth < 700;
  var sliders = JSON.parse(JSON.stringify(gd.layout.sliders || []));
  var upd = {sliders: [], 'margin.l': 0, 'margin.r': 0, 'margin.b': 0, 'margin.t': small ? 30 : 40};
  upd.showlegend = !small;
  if (small) {
    upd['title.font.size'] = 11;
    upd['legend.font.size'] = 9;
  }
  Plotly.relayout(gd, upd);
  sliders.forEach(function (s) {
    var row = document.createElement('div'); row.className = 'row';
    var lab = document.createElement('span');
    var inp = document.createElement('input');
    inp.type = 'range'; inp.min = 0; inp.max = s.steps.length - 1; inp.step = 1;
    inp.value = s.active || 0;
    var prefix = (s.currentvalue && s.currentvalue.prefix) || '';
    function show(i){ lab.textContent = prefix + s.steps[i].label; }
    show(+inp.value);
    inp.addEventListener('input', function () {
      var st = s.steps[+inp.value];
      Plotly[st.method].apply(null, [gd].concat(st.args));
      show(+inp.value);
    });
    row.appendChild(lab); row.appendChild(inp); ctl.insertBefore(row, ctl.lastElementChild);
  });
  if (small) {
    var surf = [];
    gd.data.forEach(function (t, i) { if (t.type === 'surface') surf.push(i); });
    if (surf.length) Plotly.restyle(gd, {'colorbar.thickness': 8, 'colorbar.len': 0.4,
      'colorbar.tickfont.size': 9, 'colorbar.title.font.size': 10, 'colorbar.x': 0.98}, surf);
  }
  Plotly.Plots.resize(gd);                       // fit the space left after the control bar
  window.addEventListener('resize', function () { Plotly.Plots.resize(gd); });
  document.getElementById('leg').addEventListener('click', function () {
    Plotly.relayout(gd, {showlegend: !gd.layout.showlegend});
  });
})();
</script>
</body>
</html>
"""

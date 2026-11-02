import matplotlib.pyplot as plt
# (label, low, high, category)  W/m^2 ; point values use low==high
D = [
 ("Continental background", 0.065, 0.065, "bg"),
 ("Hawaiian swell (excess)", 0.007, 0.012, "rift"),
 ("Malawi Rift (incl. Rungwe)", 0.053, 0.12, "rift"),
 ("Rio Grande Rift / San Juan", 0.107, 0.17, "rift"),
 ("Amundsen Sea / Thwaites", 0.05, 0.23, "ant"),
 ("Iceland volcanic zone", 0.215, 0.56, "rift"),
 ("NEGIS onset (proposed)", 0.97, 0.97, "ant"),
 ("Yellowstone caldera avg.", 1.5, 2.2, "volc"),
 ("Katla caldera (long-term)", 2.0, 4.0, "volc"),
 ("Takahe trough (this study)", 2.44, 4.4, "me"),
 ("Yellowstone Lake hot spots", 5.0, 15.6, "volc"),
 ("Yellowstone steam basins", 30, 35, "volc"),
 ("Eruption (Katla 1999)", 1e4, 1e5, "erupt"),
]
C = {"bg":"#9AA0A6","rift":"#4FA3D9","ant":"#7FD1C7","volc":"#F28E2B","erupt":"#E15759","me":"#FFC000"}
W = "white"
plt.rcParams.update({"font.size":15,"text.color":W,"axes.labelcolor":W,"xtick.color":W,"ytick.color":W,"axes.edgecolor":W,"font.family":"serif"})
fig, ax = plt.subplots(figsize=(13,7.5))
fig.patch.set_alpha(0); ax.patch.set_alpha(0)
for i,(lab,lo,hi,c) in enumerate(D):
    y = len(D)-1-i
    me = c=="me"
    if me:
        ax.axhspan(y-0.45,y+0.45,color=C[c],alpha=0.18,zorder=0)
    if hi>lo:
        ax.plot([lo,hi],[y,y],color=C[c],lw=12 if me else 9,solid_capstyle="round",zorder=3)
    else:
        ax.plot(lo,y,"o",color=C[c],ms=13,zorder=3)
    if me:
        ax.plot(2.44,y,"D",color="k",mfc=C[c],ms=12,zorder=4); ax.plot(4.4,y,"D",color="k",mfc=C[c],ms=12,zorder=4)
        ax.text(5.4,y,"median 2.44 · peak 4.4 W m$^{-2}$",va="center",color=C[c],fontsize=14,weight="bold")
ax.axvline(0.1,color=W,ls="--",lw=1,alpha=0.6)
ax.text(0.105,len(D)-0.6,"steady conduction limit ~0.1",fontsize=11,alpha=0.8,va="top")
ax.set_yticks(range(len(D))); ax.set_yticklabels([d[0] for d in D][::-1])
for t,d in zip(ax.get_yticklabels(),D[::-1]):
    if d[3]=="me": t.set_color(C["me"]); t.set_fontweight("bold")
ax.set_xscale("log"); ax.set_xlim(4e-3,3e5); ax.set_ylim(-0.6,len(D)-0.4)
ax.set_xlabel("Heat flux (W m$^{-2}$)")
ax.grid(axis="x",which="major",color=W,alpha=0.2)
for s in ("top","right"): ax.spines[s].set_visible(False)
from matplotlib.lines import Line2D
h=[Line2D([],[],color=C[k],lw=8,label=l) for k,l in [("bg","Background"),("rift","Rifts / plumes"),("ant","Antarctic / Greenland"),("volc","Active calderas"),("erupt","Eruption")]]
lg=ax.legend(handles=h,loc="upper right",frameon=False,fontsize=12)
fig.tight_layout()
fig.savefig("Takahe_heatflux_comparison.png",dpi=250,transparent=True)

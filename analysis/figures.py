"""Draw the principal and supplementary SCLARA figures from explicit scientific inputs."""

from pathlib import Path
import argparse, json, hashlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from analysis.dates import overlaps_period

MM = 1 / 25.4
INK = "#222222"
BLUE = "#21618c"
ORANGE = "#bc601b"
TEAL = "#137b72"
GREY = "#767676"
LIGHT = "#edf1f3"
COLORS = {
    "I1_L22": "#21618c",
    "I1_Z58": "#137b72",
    "I1_other": "#79b4ce",
    "R1a_Z284": "#b04d82",
    "R1b_U106": "#bc601b",
    "R1b_P312": "#d9ac43",
    "N": "#647c38",
    "Other": "#767676",
    "I1": "#21618c",
    "other_or_unresolved": "#999999",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 9,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)
P = ["Late_Neolithic", "Bronze_Age", "Roman_Iron_Age", "Migration"]
PL = ["Late Neolithic", "Bronze Age", "Roman Iron Age", "Migration"]
JOBS = ["linear_0", "linear_100000", "quadratic_0", "quadratic_100000"]
JC = [BLUE, TEAL, ORANGE, GREY]
JM = ["o", "s", "^", "D"]
FAMILY = [
    "Building traces",
    "Combustion",
    "Pits / deposits",
    "Heated stone",
    "Clearance / cultivation",
    "Burial",
    "Water",
]


def panel(fig, x, y, label, title):
    fig.text(x, y, label, fontsize=11, weight="bold", va="top")
    fig.text(x + 0.033, y, title, fontsize=9.2, weight="bold", va="top")


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5)
    return ax


def box(ax, x, y, w, h, title, body, color=LIGHT, size=8.7):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.008",
            facecolor=color,
            edgecolor="#b8c2c8",
            lw=0.65,
        )
    )
    ax.text(x + 0.018, y + h - 0.025, title, weight="bold", fontsize=9.2, va="top")
    ax.text(x + 0.018, y + h - 0.078, body, fontsize=size, va="top", linespacing=1.35)


def arrow(ax, a, b):
    ax.annotate(
        "",
        xy=b,
        xytext=a,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1,
            "color": GREY,
            "shrinkA": 4,
            "shrinkB": 4,
        },
    )


def interval(ax, row, y, color=BLUE, marker="o", label=None, open_marker=False, lw=1.2):
    ax.plot([row["lower"], row["upper"]], [y, y], color=color, lw=lw, zorder=3)
    ax.plot(
        row["mean"],
        y,
        marker=marker,
        ms=4,
        mec=color,
        mfc="white" if open_marker else color,
        ls="none",
        label=label,
        zorder=4,
    )


def base_map(ax, e, grey=True):
    for b in e["boundaries"]:
        a = np.array(b["xy_km"])
        ax.add_patch(
            Polygon(
                a,
                closed=True,
                facecolor="#f1f1ef" if b["country"] == "Sweden" else "#fafafa",
                edgecolor="#9ca3a6",
                lw=0.35,
                zorder=0,
            )
        )
    x0, x1, y0, y1 = e["map_limits_km"]
    ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal")
    ax.set_axis_off()


def scalebar(ax):
    ax.plot([740, 940], [6190, 6190], color=INK, lw=1)
    ax.text(840, 6230, "200 km", ha="center", fontsize=8)


def finish(fig, out, n):
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_fontsize() < 8.4:
            text.set_fontsize(8.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():
            continue
        bb = t.get_window_extent(renderer)
        w, h = fig.canvas.get_width_height()
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > w + 0.5 or bb.y1 > h + 0.5:
            outside.append(t.get_text())
    fig.savefig(
        out / f"Figure_{n}.pdf",
        metadata={
            "Title": f"SCLARA Figure {n}",
            "Author": "",
            "Creator": "Matplotlib",
            "CreationDate": None,
        },
        dpi=600,
    )
    fig.savefig(out / f"Figure_{n}.png", dpi=600)
    receipt = {
        "figure": n,
        "width_mm": round(fig.get_figwidth() / MM, 3),
        "height_mm": round(fig.get_figheight() / MM, 3),
        "outside_text": outside,
        "pdf_sha256": hashlib.sha256(
            (out / f"Figure_{n}.pdf").read_bytes()
        ).hexdigest(),
    }
    plt.close(fig)
    return receipt


def figure1(e, a, out):
    fig = plt.figure(figsize=(174 * MM, 167 * MM))
    ax = fig.add_axes([0.04, 0.02, 0.92, 0.95])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(
        0,
        1,
        "a  From source assertions to a historical question",
        weight="bold",
        fontsize=10,
        va="top",
    )
    box(
        ax,
        0.01,
        0.76,
        0.45,
        0.18,
        "Modern paternal evidence",
        "Reported origins → reconciled units\nBranch paths and multiple localities\nSampling and dependence retained",
    )
    box(
        ax,
        0.54,
        0.76,
        0.45,
        0.18,
        "Dated archaeological evidence",
        "Context, material, date and geometry\nOriginal assertions + source corrections\nPeriod and feature support checked",
    )
    arrow(ax, (0.23, 0.76), (0.37, 0.68))
    arrow(ax, (0.76, 0.76), (0.62, 0.68))
    box(
        ax,
        0.13,
        0.51,
        0.74,
        0.17,
        "A common measurement and explicit contrasts",
        "Nearest-feature proximity → fixed archaeological basis\nUnit-level locality mixture → branch and direct period contrasts\nSeparate profiles retain their separate archaeological meanings",
    )
    # Ancient evidence is an independent input to appraisal.
    box(
        ax,
        0.01,
        0.245,
        0.45,
        0.19,
        "Dated people and activity context",
        "Ancient genomes supply dated people\nActivities checked against reports\nDates, low coverage and kinship visible",
    )
    box(
        ax,
        0.54,
        0.245,
        0.45,
        0.19,
        "Historical appraisal",
        "Formation, redistribution or later sorting?\nAssess agreement, alternatives and gaps\nModern concordance does not date descent",
    )
    arrow(ax, (0.46, 0.34), (0.54, 0.34))
    arrow(ax, (0.5, 0.51), (0.75, 0.435))
    ax.text(
        0,
        0.197,
        "b  Mannaberg: one context enters a landscape measurement",
        weight="bold",
        fontsize=10,
        va="top",
    )
    x = e["worked_example"]
    v = x["full_mean_family_proximities"][4]
    ax.text(
        0.01,
        0.151,
        "Reported cultivation context\nA1462 · ash charcoal · Ua-55701\n3449 ± 30 BP; 1880–1680 BCE\nCrop remains are not directly dated",
        va="top",
        fontsize=8.4,
        linespacing=1.35,
    )
    ax.text(
        0.51,
        0.151,
        f'Closest represented nominal origin: {x["distance_km"]:.2f} km\nSite proximity exp(−d / 50) = {x["nominal_site_proximity"]:.3f}\nAll-site mean clearance proximity = {v:.3f}\nFull seven-family Component 3 = {x["full_Late_Neolithic_PC3"]:.3f}',
        va="top",
        fontsize=8.4,
        linespacing=1.35,
    )
    ax.text(
        0.01,
        0.017,
        "The numerical example uses all eligible sites and paired date/location draws for the full score.\nA dated ash sample qualifies the context; it does not identify the occupants’ paternal lineage.",
        va="bottom",
        fontsize=8,
        color=GREY,
    )
    return finish(fig, out, 1)


def figure2(e, a, out):
    fig = plt.figure(figsize=(174 * MM, 218 * MM))
    panel(fig, 0.04, 0.98, "a", "Reported paternal origins and lineages")
    ax = fig.add_axes([0.015, 0.48, 0.38, 0.46])
    base_map(ax, e)
    xy = a["origin_xy_km"]
    local = a["local_indices"]
    ys = a["local_categories"]
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c=INK, alpha=0.55, lw=0, rasterized=True)
    scalebar(ax)
    ax = fig.add_axes([0.47, 0.755, 0.48, 0.17])
    clean(ax)
    counts = e["cohort"]["category_counts"]
    ax.barh(
        np.arange(8), counts, color=[COLORS[c] for c in e["categories"]], height=0.67
    )
    ax.set_yticks(
        np.arange(8),
        [
            "I1-L22",
            "I1-Z58",
            "Other I1",
            "R1a-Z284",
            "R1b-U106",
            "R1b-P312",
            "N",
            "Other",
        ],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Represented paternal units")
    ax.set_xlim(0, 285)
    for i, n in enumerate(counts):
        ax.text(n + 5, i, str(n), va="center", fontsize=8)
    fig.text(
        0.47,
        0.705,
        "1,335 units · 1,594 locality rows\n1,399 fitted positions · 246 blocks\nMeasurement inventory: 1,404 positions",
        fontsize=8.2,
        va="top",
        linespacing=1.35,
    )
    panel(fig, 0.43, 0.64, "b", "The local distance measurement")
    from pyproj import Transformer

    example = e["worked_example"]
    transform = Transformer.from_crs(4326, 3006, always_xy=True)
    site = np.asarray([example["site"]["easting"], example["site"]["northing"]]) / 1000
    origin = np.asarray(transform.transform(*example["position_lon_lat"])) / 1000
    middle = (site + origin) / 2
    ax = fig.add_axes([0.43, 0.465, 0.24, 0.15])
    base_map(ax, e)
    ax.set(xlim=(middle[0] - 6, middle[0] + 6), ylim=(middle[1] - 6, middle[1] + 6))
    ax.plot([site[0], origin[0]], [site[1], origin[1]], color=GREY, lw=1)
    ax.scatter(
        *site, marker="s", s=28, facecolor="white", edgecolor=INK, lw=1, zorder=3
    )
    ax.scatter(*origin, marker="o", s=23, color=INK, lw=0, zorder=3)
    ax.text(site[0] - 0.8, site[1] + 0.8, "Site", ha="right", fontsize=8)
    ax.text(origin[0] + 0.7, origin[1] - 1.1, "Origin", ha="left", fontsize=8)
    ax.plot([middle[0] - 4.5, middle[0] - 2.5], [middle[1] - 4.5] * 2, color=INK, lw=1)
    ax.text(middle[0] - 3.5, middle[1] - 4, "2 km", ha="center", fontsize=8)
    fig.text(
        0.70,
        0.60,
        f'Mannaberg example\nSpherical distance: {example["distance_km"]:.2f} km\nexp(−d / 50) = {example["nominal_site_proximity"]:.3f}\n\nNominal points only;\nfull profile uses paired\ndate/location draws.',
        fontsize=8,
        va="top",
        linespacing=1.35,
    )
    panel(
        fig,
        0.04,
        0.45,
        "c",
        "Archaeological coverage at the qualifying site threshold",
    )
    for i, m in enumerate(e["archaeology_maps"]):
        ax = fig.add_axes([0.04 + i * 0.24, 0.235, 0.21, 0.175])
        base_map(ax, e)
        v = np.array(m["xy_km"])
        ax.scatter(
            v[:, 0], v[:, 1], s=2.5, color=INK, alpha=0.55, lw=0, rasterized=True
        )
        ax.set_title(PL[i], fontsize=8.5, pad=1)
    panel(fig, 0.04, 0.22, "d", "Site and county counts")
    ax = fig.add_axes([0.32, 0.04, 0.63, 0.137])
    support = {(x["period"], x["family"]): x for x in e["support"]}
    ax.set(xlim=(-.5, 3.5), ylim=(6.5, -.5))
    ax.set_xticks(range(4), ["Late\nNeolithic", "Bronze\nAge", "Roman\nIron Age", "Migration"])
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, pad=5)
    ax.set_yticks(range(7), FAMILY)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    for i, f in enumerate(e["families"]):
        for j, p in enumerate(P):
            r = support[p, f]
            ax.text(
                j,
                i,
                f'{r["sites"]} / {len(r["counties"])}',
                ha="center",
                va="center",
                color=INK,
                fontsize=8,
            )
    fig.text(
        0.04,
        0.012,
        "Qualifying probability ≥ 0.1; ≥ 10 sites in ≥ 3 counties per family. Spatial coverage still differs.",
        fontsize=8,
    )
    return finish(fig, out, 2)


def figure3(e, a, out):
    """Source contexts, fixed measurement and early lineage differentiation."""
    fig = plt.figure(figsize=(174 * MM, 220 * MM))
    panel(fig, 0.04, 0.985, "a", "Mannaberg activities in the original report")
    for left, color, title, body, oven in [
        (0.04, TEAL, "A1462 · Probable cultivation horizon", "Ash charcoal · 3449 ± 30 BP\n1880–1680 BCE (reported)\nCereal grain is undated", False),
        (0.53, ORANGE, "A645 · Cereal-rich cooking oven", "Oak charcoal · 3667 ± 44 BP\n2200–1920 BCE (reported)\nLater pottery; repeated use", True),
    ]:
        ax = fig.add_axes([left, 0.80, 0.43, 0.14]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
        ax.text(0,0.98,title,color=color,weight="bold",va="top",fontsize=8.4)
        if oven:
            ax.add_patch(Polygon([(0.025,.29),(.025,.53),(.06,.67),(.14,.71),(.21,.64),(.235,.49),(.235,.29)],closed=True,facecolor="#f4e8d9",edgecolor=color,lw=1.2))
            ax.plot([.025,.235],[.29,.29],c=color,lw=2)
            ax.scatter([.08,.11,.15,.18],[.40,.49,.38,.48],s=12,marker="o",c=color)
        else:
            ax.add_patch(Polygon([(.015,.32),(.245,.32),(.245,.60),(.015,.60)],facecolor="#dbeae4",edgecolor=color,lw=1))
            ax.plot([.015,.245],[.32,.32],c=color,lw=2)
            ax.scatter([.06,.11,.17,.21],[.42,.49,.39,.52],s=9,c=color)
        ax.text(.29,.70,body,va="top",fontsize=8.4,linespacing=1.5)
        ax.text(.13,.09,"Schematic",ha="center",color=GREY,fontsize=8.4)
    panel(fig, .04,.755,"b","Frozen landscape balance")
    panel(fig, .53,.755,"c","Contributions at the example origin")
    short=["Building","Combustion","Pits / deposits","Heated stone","Clearance","Burial","Water"]
    v=np.asarray(e["basis"]["loadings"])[:,2]
    colors=[TEAL if x>0 else ORANGE for x in v]
    terms=np.array([e["worked_example"]["PC3_feature_contributions"][f] for f in e["families"]])
    for rect, values, xlabel, limits, ticks in [
        ([.265,.54,.205,.17],v,"Component 3 loading",(-.8,.8),[-.5,0,.5]),
        ([.755,.54,.205,.17],terms,"Contribution to score",(-.4,1.05),[0,.5,1]),
    ]:
        ax=fig.add_axes(rect);clean(ax);ax.barh(range(7),values,color=colors,height=.62)
        ax.set_yticks(range(7),short);ax.invert_yaxis();ax.axvline(0,c=GREY,lw=.6)
        ax.set(xlim=limits,xticks=ticks);ax.set_xlabel(xlabel,labelpad=2)
    fig.text(.54,.719,f'Example score = {terms.sum():.3f}',fontsize=8.4)
    panel(fig,.04,.44,"d","Late Neolithic landscape")
    ax=fig.add_axes([.045,.10,.31,.305]);base_map(ax,e)
    used=np.unique(a["local_indices"]);xy=a["origin_xy_km"][used]
    sc=ax.scatter(xy[:,0],xy[:,1],c=a["common_scores"][used,0,2],s=5,cmap="BrBG",norm=TwoSlopeNorm(0,-3,3),lw=0,rasterized=True)
    scalebar(ax)
    cb=fig.colorbar(sc,cax=fig.add_axes([.08,.075,.24,.012]),orientation="horizontal",extend="both")
    cb.set_ticks([-3,0,3]);cb.set_label("Reference-scaled score",labelpad=4)
    from pyproj import Transformer
    point = Transformer.from_crs(4326, 3006, always_xy=True).transform(*e["worked_example"]["position_lon_lat"])
    ax.scatter(point[0]/1000, point[1]/1000, marker="*", s=55, facecolors="white", edgecolors=INK, linewidths=.8, zorder=7)
    ax.annotate("Example\norigin", (point[0]/1000, point[1]/1000), xytext=(-10, 14), textcoords="offset points", ha="right", fontsize=8.4, arrowprops={"arrowstyle":"-", "lw":.5, "color":INK})
    fig.text(.54,.469,"Low relative proximity to heated stone\nraises the example score.",fontsize=8.4,linespacing=1.25)
    panel(fig,.43,.44,"e","National lineage contrasts")
    ax=fig.add_axes([.63,.245,.33,.160]);clean(ax)
    cs=["I1_vs_R1a_Z284","R1a_Z284_vs_R1b_U106","I1_vs_R1b_U106"]
    for j,job in enumerate(JOBS):
        for i,c in enumerate(cs):interval(ax,e["posterior"][job]["Late_Neolithic__PC3__"+c],14-6*i-j,JC[j],JM[j])
    ax.axvline(0,c=GREY,lw=.7,ls="--");ax.set_yticks([12.5,6.5,.5],["I1–Z284","Z284–U106","I1–U106"])
    ax.set(ylim=(-2,15),xlim=(-1.15,1.05),xticks=[-1,0,1]);ax.set_xlabel("Posterior contrast (95% interval)",labelpad=2)
    handles=[Line2D([],[],c=c,marker=m,lw=0,label=l,ms=4) for c,m,l in zip(JC,JM,["Linear / original","Linear / shifted","Quadratic / original","Quadratic / shifted"])]
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(.71,.182),ncol=2,frameon=False,fontsize=8.4,columnspacing=.8,handletextpad=.3)
    panel(fig,.43,.15,"f","I1–Z284 dependence sensitivity")
    ax=fig.add_axes([.63,.075,.33,.045]);clean(ax)
    key="Late_Neolithic__PC3__I1_vs_R1a_Z284"
    for j,origin in enumerate(["0","100000"]):
        q=e["sensitivity"]["crossed_dependence_"+origin][key]
        interval(ax,dict(mean=q["median"],lower=q["q025"],upper=q["q975"]),1-j,JC[j],"s",open_marker=True)
    ax.axvline(0,c=GREY,ls="--",lw=.6);ax.set_yticks([1,0],["Original","Shifted"])
    ax.set(ylim=(-.5,1.5),xlim=(-.6,1.5),xticks=[0,.5,1]);ax.set_xlabel("Crossed-weight median and range",labelpad=2)
    fig.text(.04,.012,"Contexts explain feature families; the regional score combines all seven. Nordin (2023), pp. 19–20, 67, 97, 100.",fontsize=8.4)
    return finish(fig,out,3)


def figure4(e, a, out):
    fig = plt.figure(figsize=(174 * MM, 234 * MM))
    cols = [0.045, 0.285, 0.525, 0.765]
    w = 0.19
    used = np.unique(a["local_indices"])
    xy = a["origin_xy_km"][used]
    for x, name, b in zip(cols, PL, e["period_bounds"]):
        dates = f"{abs(b[0])}–{abs(b[1])} BCE" if b[1] < 0 else f"AD {b[0]}–{b[1]}"
        fig.text(
            x + w / 2, 0.985, name, weight="bold", ha="center", va="top", fontsize=9
        )
        fig.text(x + w / 2, 0.966, dates, ha="center", va="top", fontsize=8)
    fig.text(0.04, 0.936, "a  Dated ancient observations", weight="bold", fontsize=9)
    for i, (x, b) in enumerate(zip(cols, e["period_bounds"])):
        ax = fig.add_axes([x, 0.757, w, 0.175])
        base_map(ax, e)
        ax.scatter(xy[:, 0], xy[:, 1], s=0.45, color="#dddddd", lw=0, rasterized=True)
        selected = []
        for r in e["ancient_observations"]:
            if r.get("country") != "Sweden" or "xy_km" not in r:
                continue
            bounds = [
                v["calendar_bounds"]
                for v in r["date_variants"]
                if v.get("calendar_bounds")
            ]
            if not any(overlaps_period(v, b[0], b[1]) for v in bounds):
                continue
            selected.append(r)
            c = BLUE if r["plot_branch"].startswith("I1") else COLORS[r["plot_branch"]]
            xx, yy = r["xy_km"]
            context = "context" in str(r["date_basis"]).lower()
            qualification = r["structured_qualification"]
            flag = qualification["quality_caveat"] or bool(qualification["date_flags"])
            excluded = qualification["any_source_population_exclusion"]
            ax.scatter(
                [xx],
                [yy],
                s=13 if r["plot_branch"] != "other_or_unresolved" else 6,
                marker=(
                    "x"
                    if flag or excluded
                    else {"R1b_U106": "s", "R1b_P312": "^", "R1a_Z284": "D"}.get(
                        r["plot_branch"], "o"
                    )
                ),
                edgecolors=c if not (flag or excluded) else None,
                facecolors="none" if context and not (flag or excluded) else c,
                lw=0.7,
                zorder=3,
            )
        ax.text(
            0.5,
            -0.04,
            f"{len(selected)} plotted identities",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
        )
        if i < 2:
            east = sum(
                r.get("pre2800BP_cluster") == "pre2.8k_0_1_3_2-EScan" for r in selected
            )
            south = sum(
                r.get("pre2800BP_cluster") == "pre2.8k_0_1_3_1-SScan" for r in selected
            )
            ax.text(
                0.5,
                0.92,
                f"Ancestry E/S: {east}/{south}",
                transform=ax.transAxes,
                ha="center",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
            )
    handles = [
        Line2D(
            [0],
            [0],
            marker={"R1b_U106": "s", "R1b_P312": "^", "R1a_Z284": "D"}.get(k, "o"),
            color="none",
            mfc=COLORS[k],
            mec=COLORS[k],
            ms=3.5,
            label=l,
        )
        for k, l in [
            ("I1", "I1 incl. subdivisions"),
            ("R1b_U106", "U106"),
            ("R1b_P312", "P312"),
            ("R1a_Z284", "Z284"),
            ("other_or_unresolved", "Other / unresolved"),
        ]
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.747),
        ncol=5,
        fontsize=8,
        frameon=False,
        columnspacing=0.9,
        handlelength=0.6,
    )
    fig.text(
        0.04,
        0.712,
        "Open: contextual date; ×: specified quality caveat or source exclusion. E/S: EScan/SScan source clusters.\nDate spans can cross periods. Grey points: reported paternal origins. Identities are not independent lineages.",
        fontsize=8,
        va="top",
        linespacing=1.35,
    )
    fig.text(
        0.04,
        0.657,
        "b  The same archaeological balance at the same reported origins",
        weight="bold",
        fontsize=9,
    )
    for i, x in enumerate(cols):
        ax = fig.add_axes([x, 0.48, w, 0.16])
        base_map(ax, e)
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=a["common_scores"][used, i, 2],
            s=2.8,
            cmap="BrBG",
            norm=TwoSlopeNorm(0, -3, 3),
            lw=0,
            rasterized=True,
        )
    cb = fig.colorbar(
        sc,
        cax=fig.add_axes([0.37, 0.468, 0.26, 0.009]),
        orientation="horizontal",
        extend="both",
    )
    cb.set_ticks([-3, 0, 3])
    cb.ax.tick_params(labelsize=8, pad=1)
    fig.text(0.04, 0.468, "Component 3", fontsize=8)
    fig.text(0.72, 0.468, "One fixed measurement", fontsize=8)
    fig.text(
        0.04,
        0.433,
        "c  I1–Z284 contrasts on the common measurement",
        weight="bold",
        fontsize=9,
    )
    for i, x in enumerate(cols):
        ax = fig.add_axes([x, 0.361, w, 0.059])
        clean(ax)
        ax.axvline(0, color=GREY, lw=0.6, ls="--")
        for j, job in enumerate(JOBS):
            interval(
                ax,
                e["posterior"][job][P[i] + "__PC3__I1_vs_R1a_Z284"],
                3 - j,
                JC[j],
                JM[j],
            )
        ax.set(yticks=[], ylim=(-0.6, 3.6), xlim=(-0.8, 1.1), xticks=[-0.5, 0, 0.5, 1])
        ax.tick_params(axis="x", pad=1)
    fig.text(
        0.04,
        0.317,
        "Conditional 95% posterior intervals; four specifications as in Fig. 3.",
        fontsize=8,
    )
    fig.text(0.04, 0.288, "d  Direct period differences", weight="bold", fontsize=9)
    ax = fig.add_axes([0.24, 0.133, 0.34, 0.135])
    clean(ax)
    for j, job in enumerate(JOBS):
        for i, p in enumerate(P[1:]):
            interval(
                ax,
                e["posterior"][job][
                    "Late_Neolithic_minus_" + p + "__PC3__I1_vs_R1a_Z284"
                ],
                14 - 6 * i - j,
                JC[j],
                JM[j],
            )
    ax.set_yticks([12.5, 6.5, .5], ["LN − Bronze", "LN − Roman", "LN − Migration"])
    ax.set(ylim=(-2, 15), xlim=(-0.5, 1.5), xticks=[0, 0.5, 1, 1.5])
    ax.axvline(0, c=GREY, ls="--", lw=0.6)
    fig.text(
        0.64,
        0.29,
        "LN–Roman: conditional posterior\nintervals exclude zero.\nCrossed-dependence ranges cross zero.\nLN–Migration lower limits lie near zero.\nRoman–Migration remains unresolved.",
        fontsize=8,
        va="top",
        linespacing=1.4,
    )
    # Give the comparable early periods their own page, preserving physical scale.
    floor = 0.105
    for ax in fig.axes:
        x, y, w, h = ax.get_position().bounds
        ax.set_position([x, (y - floor) / (1 - floor), w, h / (1 - floor)])
    for text in fig.texts:
        x, y = text.get_position()
        text.set_position((x, (y - floor) / (1 - floor)))
    for legend in fig.legends:
        legend.set_bbox_to_anchor((0.50, (0.747-floor)/(1-floor)), transform=fig.transFigure)
    fig.set_size_inches(174 * MM, 234 * (1-floor) * MM)
    return finish(fig, out, 4)


def ancient_map(ax, e, bounds):
    """Plot eligible source observations; coincident people remain visible in counts."""
    base_map(ax, e)
    rows = [
        r
        for r in e["ancient_observations"]
        if "xy_km" in r
        and any(
            overlaps_period(v.get("calendar_bounds"), *bounds)
            for v in r["date_variants"]
        )
    ]
    for r in rows:
        q = r["structured_qualification"]
        branch = r["plot_branch"]
        color = BLUE if branch.startswith("I1") else COLORS[branch]
        flag = (
            q["quality_caveat"]
            or q["any_source_population_exclusion"]
            or bool(q["date_flags"])
        )
        marker = (
            "x"
            if flag
            else {"R1b_U106": "s", "R1b_P312": "^", "R1a_Z284": "D"}.get(branch, "o")
        )
        context = "context" in str(r["date_basis"]).lower()
        ax.scatter(
            *r["xy_km"],
            s=16 if branch != "other_or_unresolved" else 7,
            marker=marker,
            facecolors="none" if context and marker != "x" else color,
            edgecolors=color,
            linewidths=0.6,
        )
    return rows


def figure5(e, a, out):
    fig = plt.figure(figsize=(174 * MM, 211 * MM))
    periods = [
        ("Viking", [750, 1050], "viking_primary"),
        ("Medieval", [1050, 1520], "medieval_primary"),
    ]
    panel(fig, 0.04, 0.985, "a", "Ancient locations and later activity landscapes")
    ax=fig.add_axes([.045,.667,.24,.255])
    rows=ancient_map(ax,e,[750,1520]);ax.set_title("Ancient locations",fontsize=9)
    ax.text(.5,-.045,f"{len(rows)} distinct identities",ha="center",transform=ax.transAxes,fontsize=8.4)
    for j,(title,bounds,job) in enumerate(periods):
        ax=fig.add_axes([.38+j*.33,.667,.24,.255]);base_map(ax,e)
        sc=ax.scatter(a["origin_xy_km"][:,0],a["origin_xy_km"][:,1],c=a["later_scores"][:,e["later_jobs"].index(job)],s=2.5,cmap="BrBG",norm=TwoSlopeNorm(0,-3,3),lw=0,rasterized=True)
        ax.set_title(title+" profile",fontsize=9)
    cb = fig.colorbar(
        sc,
        cax=fig.add_axes([0.38, 0.642, 0.24, 0.009]),
        orientation="horizontal",
        extend="both",
    )
    cb.set_ticks([-3, 0, 3])
    cb.ax.tick_params(pad=1)
    fig.text(
        0.04,
        0.571,
        "Viking AD 750–1050 · Medieval AD 1050–1520. One ancient map shows the union of both windows.\nThe two profile maps have distinct archaeological meanings and reference units.",
        fontsize=8.4,
        linespacing=1.4,
    )
    panel(fig, 0.04, 0.535, "b", "Branch contrasts and dependence sensitivity")
    specifications = [
        (
            "viking_primary",
            ["I1_L22_vs_I1_Z58", "R1b_U106_vs_R1b_P312"],
            ["L22 − Z58", "U106 − P312"],
        ),
        (
            "medieval_primary",
            ["R1a_Z284_vs_R1b_U106", "R1b_U106_vs_R1b_P312"],
            ["Z284 − U106", "U106 − P312"],
        ),
    ]
    for j, (job, names, labels) in enumerate(specifications):
        ax = fig.add_axes([0.21 + j * 0.49, 0.36, 0.255, 0.132])
        clean(ax)
        ax.set_title("Viking" if j == 0 else "Medieval", fontsize=9, pad=6)
        for i, name in enumerate(names):
            interval(ax, e["later_posterior"][job][name], 1 - i + 0.1, BLUE)
            q = e["later_sensitivity"][job][name]
            interval(
                ax,
                dict(mean=q["median"], lower=q["q025"], upper=q["q975"]),
                1 - i - 0.1,
                GREY,
                "s",
                open_marker=True,
            )
        ax.set_yticks([1, 0], labels)
        ax.set(xlim=(-1.3, 1.1), ylim=(-0.5, 1.5), xticks=[-1, 0, 1])
        ax.axvline(0, c=GREY, ls="--", lw=0.6)
    fig.text(
        0.04,
        0.307,
        "Filled circles: posterior means and 95% intervals. Open squares: crossed-weight medians and ranges.\nEstimates use the original branch representation; coarser-cut sensitivities are reported separately.",
        fontsize=8.4,
        linespacing=1.4,
    )
    panel(fig, 0.04, 0.264, "c", "Original contexts define both poles")
    fig.text(
        0.04,
        0.225,
        "Viking activity balance\nPositive: charcoal production · Färila\nPositive: iron-working waste · Harbo-Smedsbo\nNegative: hearth · Fyrislund A6868\nRelative to recorded posthole/hearth settings.",
        fontsize=8.4,
        va="top",
        linespacing=1.55,
    )
    fig.text(
        0.53,
        0.225,
        "Medieval activity balance\nPositive: brick kiln · Uppsala A817\nNegative: tar production · Leksand\nNegative: charcoal production · Nifsarp\nProduction occurs on both poles.",
        fontsize=8.4,
        va="top",
        linespacing=1.55,
    )
    fig.text(
        0.04,
        0.063,
        "Ancient markers: blue I1; orange square U106; gold triangle P312; magenta diamond Z284; grey unresolved.\nAll plotted medieval U106, P312 and Z284 people also overlap the Viking window.\nOpen: contextual date; ×: source quality, exclusion or date qualification.",
        fontsize=8.4,
        va="top",
        linespacing=1.4,
    )
    return finish(fig, out, 5)


def figureS1(e, a, out):
    fig = plt.figure(figsize=(174 * MM, 205 * MM))
    panel(fig, 0.04, 0.98, "a", "Regional support and the influential paternal block")
    ax = fig.add_axes([0.04, 0.47, 0.39, 0.45])
    base_map(ax, e)
    palette = {
        "SE11": "#cab2d6",
        "SE12": "#a6cee3",
        "SE21": "#b2df8a",
        "SE22": "#fdbf6f",
        "SE23": "#fb9a99",
        "SE3": "#dddddd",
    }
    for r in e["regional_shapes"]:
        ax.add_patch(
            Polygon(
                np.asarray(r["xy_km"]),
                closed=True,
                fc=palette[r["region"]],
                ec="white",
                lw=0.4,
                alpha=0.75,
                zorder=1,
            )
        )
    block = np.array(e["influential_block"]["xy_km"])
    ax.scatter(block[:, 0], block[:, 1], s=13, marker="x", color=INK, lw=0.7, zorder=4)
    for region, x, y in [
        ("SE3", 790, 7220),
        ("SE11", 690, 6600),
        ("SE12", 560, 6525),
        ("SE21", 570, 6350),
        ("SE22", 425, 6170),
        ("SE23", 340, 6460),
    ]:
        ax.text(x, y, region, ha="center", fontsize=8.4, zorder=6)
    fig.text(
        0.04,
        0.441,
        "× B-26258: 20 units; 25 distinct positions\n58.84–66.10°N; median latitude 63.18°N",
        fontsize=8.4,
        va="top",
        linespacing=1.4,
    )
    regions = ["SE11", "SE12", "SE21", "SE22", "SE23", "SE3"]
    names = [
        "Stockholm",
        "East Middle",
        "Småland and islands",
        "South",
        "West",
        "Northern groups",
    ]
    ax = fig.add_axes([0.52, 0.51, 0.43, 0.38])
    ax.axis("off")
    rows = []
    for region, name in zip(regions, names):
        r = next(
            q for q in e["regional"] if q["region"] == region and q["job"] == "linear_0"
        )
        rows.append(
            [
                region,
                str(r["units"]),
                str(r["total_blocks"]),
                str(sum(r["category_units"][:3])),
                str(r["category_units"][3]),
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=["Group", "Units", "Blocks", "I1", "Z284"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1, 1.65)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor(LIGHT)
            cell.set_text_props(weight="bold")
    fig.text(
        0.51,
        0.485,
        "SE3 combines SE31, SE32 and SE33.\n71 mixed or unassigned units are outside these\nregional restrictions; they remain in the national fit.",
        fontsize=8.4,
        linespacing=1.4,
    )
    panel(fig, 0.04, 0.37, "b", "Regional Late Neolithic I1–Z284 estimates")
    ax = fig.add_axes([0.18, 0.13, 0.74, 0.20])
    clean(ax)
    for i, region in enumerate(regions):
        for j, job in enumerate(["linear_0", "linear_100000"]):
            r = next(
                q for q in e["regional"] if q["region"] == region and q["job"] == job
            )
            v = r["values"]["Late_Neolithic__PC3__I1_vs_R1a_Z284"]
            interval(
                ax,
                dict(mean=v["mode"], lower=v["lower"], upper=v["upper"]),
                5 - i + (0.13 if j == 0 else -0.13),
                (BLUE if j == 0 else TEAL) if r["focal_display_support"] else "#aaaaaa",
                "o" if j == 0 else "s",
            )
    ax.set_yticks(range(6), regions[::-1])
    ax.axvline(0, c=GREY, ls="--", lw=0.6)
    ax.set(xlim=(-1.8, 2.1), xticks=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    ax.set_xlabel("Conditional mode and inverse-Hessian 95% interval")
    fig.text(
        0.04,
        0.068,
        "Circles: original grid; squares: shifted grid. Grey: fewer than ten units or three blocks in a constituent\nI1–Z284 category. These are six restricted cohorts, each fitted twice. Complete category-level\nunit and block counts accompany the numerical inventory.",
        fontsize=8.4,
        va="top",
        linespacing=1.4,
    )
    return finish(fig, out, "S1")


def source_benchmark_panels(e, a, out, section):
    fig = plt.figure(figsize=(174 * MM, 215 * MM))
    panel(fig, 0.04, 0.98, "a", "Date allocation and source-pool expansion")
    oldperiods = ["Pre2800", "Late_Neolithic", "Early_Bronze", "Migration"]
    labels = ["4000–2800 BCE", "2300–1700 BCE", "1700–1100 BCE", "AD 400–550"]
    for j, origin in enumerate(["0", "100000"]):
        ax = fig.add_axes([0.20 + j * 0.43, 0.675, 0.33, 0.235])
        clean(ax)
        for i, p in enumerate(oldperiods):
            target = p + "__I1_vs_R1a_Z284"
            for scenario, dy, col, mark in [
                ("fixed_original_pool", 0.20, BLUE, "o"),
                ("expanded_pool", -0.20, ORANGE, "s"),
            ]:
                r = e["paired"][scenario][origin][target]
                q = r["mode_quantiles"]
                interval(
                    ax,
                    {"mean": q[2], "lower": q[1], "upper": q[3]},
                    3 - i + dy,
                    col,
                    mark,
                )
            q = e["paired"]["fixed_original_pool"][origin][target]
            ax.scatter(
                q["strict_mode"],
                3 - i,
                marker="D",
                s=18,
                facecolors="white",
                edgecolors=INK,
                lw=0.7,
                zorder=4,
            )
        ax.set_yticks([3, 2, 1, 0], labels if j == 0 else [""] * 4)
        ax.set(ylim=(-0.5, 3.5), xlim=(-0.4, 0.6), xticks=[-0.4, 0, 0.4])
        ax.axvline(0, c=GREY, ls="--", lw=0.6)
        ax.set_title("Original grid" if j == 0 else "Shifted grid", fontsize=9)
        ax.set_xlabel("I1–Z284 conditional mode")
    handles = [
        Line2D([0], [0], c=BLUE, marker="o", label="Original pool"),
        Line2D([0], [0], c=ORANGE, marker="s", label="Expanded pool"),
        Line2D(
            [0],
            [0],
            c=INK,
            marker="D",
            mfc="white",
            lw=0,
            label="Strict original panel",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.625),
        ncol=3,
        frameon=False,
        fontsize=8,
        columnspacing=1.2,
    )
    fig.text(
        0.04,
        0.58,
        "Dots and bars: median and 2.5–97.5% range of 64 paired allocation modes, not posterior intervals.\nMigration medians: 0.349 / 0.412 within the original pool; 0.047 / 0.062 after expansion.",
        fontsize=8.2,
        va="top",
        linespacing=1.45,
    )
    panel(fig, 0.04, 0.515, "b", "Chronological discrimination")
    cases = [
        "Pre2800_0.4",
        "Late_Neolithic_0.4",
        "Early_Bronze_0.4",
        "Migration_0.4",
        "recording_Migration",
    ]
    names = [
        "4000–2800 BCE",
        "Late Neolithic",
        "Early Bronze",
        "Migration",
        "Recording stress",
    ]
    ax = fig.add_axes([0.24, 0.31, 0.25, 0.18])
    clean(ax)
    for i, s in enumerate(cases):
        r = e["benchmarks"][s]["periods"]["correct_nomination"]
        interval(
            ax,
            {
                "mean": 100 * r["estimate"],
                "lower": 100 * r["lower"],
                "upper": 100 * r["upper"],
            },
            4 - i,
            BLUE,
        )
    ax.set_yticks(range(5), names[::-1])
    ax.set(xlim=(-1.2, 22), ylim=(-0.5, 4.5), xticks=[0, 10, 20])
    ax.set_xlabel("Correct unique period (%)")
    fig.text(
        0.04,
        0.255,
        "Injected contrast = 0.4; 400 datasets per condition.\nIntervals are 95% Wilson intervals.\nZero wrong-period nominations in these conditions\ndoes not imply high power to identify the correct one.",
        fontsize=8,
        va="top",
        linespacing=1.5,
    )
    panel(fig, 0.56, 0.515, "c", "Predictive performance")
    order = [
        "null",
        "Pre2800_0.2",
        "Late_Neolithic_0.2",
        "Early_Bronze_0.2",
        "Migration_0.2",
        "Pre2800_0.4",
        "Late_Neolithic_0.4",
        "Early_Bronze_0.4",
        "Migration_0.4",
        "shared_equal",
        "pooled",
        "recording_null",
        "recording_Migration",
    ]
    short = [
        "Null",
        "Pre 0.2",
        "LN 0.2",
        "EB 0.2",
        "Mig 0.2",
        "Pre 0.4",
        "LN 0.4",
        "EB 0.4",
        "Mig 0.4",
        "Equal 0.2",
        "Pooled 0.4",
        "Stress null",
        "Stress Mig",
    ]
    ax = fig.add_axes([0.70, 0.19, 0.26, 0.30])
    clean(ax)
    for i, s in enumerate(order):
        for m, dy, c, mark in [
            ("geography", 0.12, BLUE, "o"),
            ("pooled", -0.12, ORANGE, "s"),
        ]:
            r = e["benchmarks"][s]["paired"][m]["expected_log"]
            interval(
                ax,
                {
                    "mean": 1000 * r["mean"],
                    "lower": 1000 * (r["mean"] - 1.96 * r["mcse"]),
                    "upper": 1000 * (r["mean"] + 1.96 * r["mcse"]),
                },
                12 - i + dy,
                c,
                mark,
            )
    ax.set_yticks(range(13), short[::-1])
    ax.set(xlim=(-12, 1), ylim=(-0.6, 12.6), xticks=[-10, -5, 0])
    ax.axvline(0, c=GREY, ls="--", lw=0.6)
    ax.set_xlabel("Expected log-score gain × 1000")
    fig.text(
        0.56,
        0.13,
        "Four-period model versus\n● geography     ■ pooled archaeology\nBars: ±1.96 Monte Carlo SE.\nNegative values favour the simpler model.",
        fontsize=8,
        va="top",
        linespacing=1.5,
    )
    fig.text(
        0.04,
        0.135,
        "Context matters with short-lived material\nAt Kvarnbacken, cereal supports material quality;\nit does not uniquely date building construction.\nNearby sites can substitute for one another;\nsingle-site omissions can understate joint influence.",
        fontsize=8.2,
        va="top",
        linespacing=1.45,
    )
    fig.text(
        0.04,
        0.016,
        "Benchmarks assess fixed designs; they do not calibrate profile selection or establish a historical mechanism.",
        fontsize=8,
    )
    if section == "source":
        for ax in list(fig.axes):
            if ax.get_position().y0 < .6: ax.remove()
        for t in list(fig.texts):
            if t.get_position()[1] < .55: t.remove()
        lower,upper=.50,1.0
        for legend in fig.legends:legend.set_bbox_to_anchor((.52,(.625-lower)/(upper-lower)),transform=fig.transFigure)
        name=6; height=100
    else:
        for ax in list(fig.axes):
            if ax.get_position().y0 > .6: ax.remove()
        for legend in list(fig.legends):legend.remove()
        for t in list(fig.texts):
            x,y=t.get_position()
            if y > .54 or (x < .1 and .13 < y < .14):t.remove()
            elif t.get_text() in ["b","c"]:t.set_text({"b":"a","c":"b"}[t.get_text()])
        lower,upper=0,.55
        name="S3"; height=125
    for ax in fig.axes:
        x,y,w,h=ax.get_position().bounds;ax.set_position([x,(y-lower)/(upper-lower),w,h/(upper-lower)])
    for t in fig.texts:
        x,y=t.get_position();t.set_position((x,(y-lower)/(upper-lower)))
    fig.set_size_inches(174*MM,height*MM)
    return finish(fig,out,name)


def figure6(e,a,out):
    return source_benchmark_panels(e,a,out,"source")


def figureS3(e,a,out):
    return source_benchmark_panels(e,a,out,"benchmarks")


def figure7(e, a, out):
    """All fixed current-estimator conditions; no favourable-condition selection."""
    evaluation = e["current_evaluation"]
    contract = e["current_evaluation_contract"]
    conditions = [r["name"] for r in contract["conditions"]]
    labels = [
        "Clean / null",
        "Clean / effect",
        "Alternative origins / null",
        "Alternative origins / effect",
        "Source multiplicity / null",
        "Source multiplicity / effect",
        "Block dependence / null",
        "Block dependence / effect",
    ]
    methods = contract["methods"]
    colors, markers = [BLUE, TEAL, ORANGE], ["o", "s", "^"]
    fig = plt.figure(figsize=(174 * MM, 210 * MM))
    panel(fig, 0.04, 0.98, "a", "Recovery of the direct Late Neolithic–Roman contrast")
    ax = clean(fig.add_axes([0.35, 0.66, 0.60, 0.26]))
    for j, method in enumerate(methods):
        for i, name in enumerate(conditions):
            r = evaluation[name][method]["bias"]
            mean, se = r["mean"][1], r["mcse"][1]
            interval(
                ax,
                dict(mean=mean, lower=mean - 1.96 * se, upper=mean + 1.96 * se),
                7 - i + (1 - j) * 0.19,
                colors[j],
                markers[j],
            )
    ax.set_yticks(range(8), labels[::-1])
    ax.set_ylim(-0.6, 7.6)
    ax.axvline(0, color=GREY, ls="--", lw=0.6)
    ax.set_xlabel("Bias (±1.96 Monte Carlo SE)")
    handles = [
        Line2D([], [], c=c, marker=m, lw=0, label=t)
        for c, m, t in zip(
            colors,
            markers,
            [
                "Unit probability mixture",
                "Unit-mean covariates",
                "Source-record weights",
            ],
        )
    ]
    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.51, 0.59),
        ncol=1,
        frameon=False,
        fontsize=8.4,
    )
    panel(fig, 0.04, 0.53, "b", "Conditional interval coverage")
    panel(fig, 0.54, 0.53, "c", "Paired predictive log score")
    ax = clean(fig.add_axes([0.12, 0.18, 0.34, 0.29]))
    bx = clean(fig.add_axes([0.63, 0.18, 0.32, 0.29]))
    for j, method in enumerate(methods):
        for i, name in enumerate(conditions):
            row = evaluation[name][method]
            n = row["successful"]
            rate = row["coverage"]["mean"][1]
            z = 1.96
            center = (rate + z * z / (2 * n)) / (1 + z * z / n)
            half = (
                z
                * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
                / (1 + z * z / n)
            )
            interval(
                ax,
                dict(
                    mean=rate * 100,
                    lower=(center - half) * 100,
                    upper=(center + half) * 100,
                ),
                7 - i + (1 - j) * 0.19,
                colors[j],
                markers[j],
            )
    for j, method in enumerate(methods[1:]):
        for i, name in enumerate(conditions):
            row = evaluation[name]["paired_vs_" + method]["expected_log"]
            mean, se = 1000 * row["mean"], 1000 * row["mcse"]
            interval(
                bx,
                dict(mean=mean, lower=mean - 1.96 * se, upper=mean + 1.96 * se),
                7 - i + (0.5 - j) * 0.2,
                colors[j + 1],
                markers[j + 1],
            )
    for plot in [ax, bx]:
        plot.set_yticks(range(8), [str(i) for i in range(8, 0, -1)])
        plot.set_ylim(-0.6, 7.6)
    ax.axvline(95, c=GREY, ls="--", lw=0.6)
    ax.set_xlabel("95% interval coverage (%)")
    bx.axvline(0, c=GREY, ls="--", lw=0.6)
    bx.set_xlabel("Mixture − comparator, ×1000")
    fig.text(
        0.04,
        0.12,
        "Rows 1–8 follow panel a. Null = 0; effect = 0.4; 200 datasets per condition.\n"
        "Coverage bars: 95% Wilson intervals. Prediction bars: ±1.96 Monte Carlo SE.\n"
        "Positive log-score differences favour the unit probability mixture.",
        va="top",
        fontsize=8.4,
        linespacing=1.4,
    )
    fig.text(
        0.04,
        0.025,
        "Fixed simulation design; conditional modes and inverse-Hessian intervals.\n"
        "Test outcomes use the same covariates and block effects; this is not geographical extrapolation.",
        va="bottom",
        fontsize=8.4,
        linespacing=1.3,
    )
    return finish(fig, out, 7)


def figureS2(e, a, out):
    """Display the fixed whole-unit common-recording support restriction."""
    info = e["shared_support_map"]
    fig = plt.figure(figsize=(174 * MM, 140 * MM))
    panel(fig, 0.04, 0.98, "a", "Shared local recording support")
    ax = fig.add_axes([0.045, 0.18, 0.36, 0.71])
    base_map(ax, e)
    scalebar(ax)
    positions = a["origin_xy_km"]
    for key, color, size in [
        ("other_position_indices", ORANGE, 4),
        ("retained_position_indices", TEAL, 3),
    ]:
        p = positions[info[key]]
        ax.scatter(p[:, 0], p[:, 1], s=size, c=color, alpha=0.65, lw=0, zorder=3)
    panel(fig, 0.47, 0.98, "b", "Whole units and category support")
    tax = fig.add_axes([0.47, 0.39, 0.49, 0.44])
    tax.axis("off")
    rows = []
    for i, cat in enumerate(e["categories"]):
        r = info["categories"][cat]
        label = {
            "I1_L22": "L22",
            "I1_Z58": "Z58",
            "I1_other": "Other I1",
            "R1a_Z284": "Z284",
            "R1b_U106": "U106",
            "R1b_P312": "P312",
        }.get(cat, cat)
        rows.append(
            [
                label,
                str(e["cohort"]["category_counts"][i])
                + "/"
                + str(e["shared_support_map"]["original_category_blocks"][cat]),
                str(r["units"]) + "/" + str(r["blocks"]),
            ]
        )
    t = tax.table(
        cellText=rows,
        colLabels=["Category", "Original", "Restricted"],
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(8.4)
    for (row, col), cell in t.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor(LIGHT)
            cell.get_text().set_weight("bold")
    fig.text(
        0.47,
        0.31,
        f"Cells: units / proxy dependence blocks.\n{info['units']:,} units retained; {info['omitted_units']} omitted.\nEvery outcome category passes the fixed\nten-unit / three-block support gate.",
        va="top",
        fontsize=8.4,
        linespacing=1.4,
    )
    fig.legend(
        handles=[
            Line2D([], [], c=TEAL, marker="o", ls="", label="Used by a retained unit"),
            Line2D(
                [], [], c=ORANGE, marker="o", ls="", label="Other represented position"
            ),
        ],
        loc="center",
        bbox_to_anchor=(0.51, 0.105),
        ncol=2,
        frameon=False,
        fontsize=8.4,
    )
    fig.text(
        0.04,
        0.035,
        "Every reported locality must have a site within 75 km in ≥ half the draws in both periods.\nPoints show distinct reported paternal origins; all alternative origins stay together within a unit.",
        va="bottom",
        fontsize=8.4,
        linespacing=1.3,
    )
    return finish(fig, out, "S2")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--figures", default="1,2,3,4,5,6,7,S1,S2,S3")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    e = json.loads((args.input / "evidence.json").read_text(encoding="utf-8"))
    a = np.load(args.input / "measurements.npz")
    receipts = []
    for n in args.figures.split(","):
        receipts.append(globals()["figure" + str(n)](e, a, args.output))
    checkpath = args.output / "figure_checks.json"
    prior = (
        json.loads(checkpath.read_text(encoding="utf-8")) if checkpath.exists() else []
    )
    merged = {r["figure"]: r for r in prior}
    merged.update({r["figure"]: r for r in receipts})
    checkpath.write_text(
        json.dumps([merged[k] for k in sorted(merged, key=str)], indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()

"""Export common-period comparisons and allocation sensitivity from retained results."""

from pathlib import Path
import argparse
import csv
import hashlib
import html
import io
import json
import sqlite3
import zipfile

import numpy as np

MODELS = ["linear_0", "linear_100000", "quadratic_0", "quadratic_100000"]
MODEL_LABELS = ["Linear geography · original grid", "Linear geography · shifted grid",
                "Quadratic geography · original grid", "Quadratic geography · shifted grid"]
PERIODS = ["Late_Neolithic", "Bronze_Age", "Roman_Iron_Age", "Migration"]
PERIOD_LABELS = ["Late Neolithic", "Bronze Age", "Roman Iron Age", "Migration period"]
CONTRASTS = ["I1_vs_R1a_Z284", "I1_vs_R1b_U106", "I1_L22_vs_I1_Z58",
             "R1b_U106_vs_R1b_P312", "R1a_Z284_vs_R1b_U106", "N_vs_I1", "I1_vs_R1b_P312"]
LINEAGE_LABELS = ["I1 − R1a-Z284", "I1 − R1b-U106", "I1-L22 − I1-Z58",
                  "R1b-U106 − R1b-P312", "R1a-Z284 − R1b-U106", "N − I1", "I1 − R1b-P312"]
FIELDS = {"posterior_mean": "mean", "lower_95_equal_tail": "lower",
          "upper_95_equal_tail": "upper", "posterior_median": "median", "posterior_sd": "sd",
          "posterior_probability_positive": "probability_positive", "rhat": "rhat",
          "ess_bulk": "ess_bulk", "ess_tail": "ess_tail", "mcse_mean": "mcse_mean",
          "mcse_lower": "mcse_lower", "mcse_upper": "mcse_upper",
          "mcse_probability_positive": "mcse_probability_positive", "scope": "scope"}
TARGETS = ["Late_Neolithic__PC3__I1_vs_R1a_Z284",
           "Late_Neolithic_minus_Roman_Iron_Age__PC3__I1_vs_R1a_Z284",
           "Late_Neolithic__PC3__R1a_Z284_vs_R1b_U106"]
TARGET_LABELS = ["Late Neolithic I1–Z284", "Late Neolithic–Roman I1–Z284 difference",
                 "Late Neolithic Z284–U106"]
NORMAL_975 = 1.959963984540054


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def complete_comparisons(results):
    """Retain every target and its original precision from the four posterior summaries."""
    periods = PERIODS + [f"{p}_minus_{q}" for i, p in enumerate(PERIODS) for q in PERIODS[i + 1:]]
    expected = {f"{p}__PC{c}__{k}" for p in periods for c in range(1, 4) for k in CONTRASTS}
    data, inputs = {}, {}
    for model in MODELS:
        source = results / "current" / f"{model}_contrasts.csv"
        with source.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        data[model] = {r["target"]: r for r in rows}
        if len(rows) != 210 or set(data[model]) != expected:
            raise ValueError(f"Incomplete or duplicated contrast family: {model}")
        digest = sha256(source)
        inputs[f"results/current/{source.name}"] = digest
        receipt_path = source.with_name(model + ".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not receipt["diagnostics"]["passed"] or receipt["contrasts_sha256"] != digest:
                raise ValueError(f"Retained contrast receipt failed: {model}")
        for row in rows:
            if not all(np.isfinite(float(row[k])) for k in FIELDS.values() if k != "scope"):
                raise ValueError(f"Non-finite retained comparison: {model}")
            if not float(row["lower"]) <= float(row["mean"]) <= float(row["upper"]):
                raise ValueError(f"Invalid retained interval: {model}")
    output, reading, overview = [], [], []
    for component in ["PC1", "PC2", "PC3"]:
        for period in periods:
            parts = period.split("_minus_")
            kind = "within_period" if len(parts) == 1 else "direct_period_difference"
            label = " minus ".join(PERIOD_LABELS[PERIODS.index(p)] for p in parts)
            for contrast, lineage_label in zip(CONTRASTS, LINEAGE_LABELS):
                target = f"{period}__{component}__{contrast}"
                base = dict(component=component, comparison_type=kind,
                            period_comparison=label, lineage_contrast=lineage_label)
                wide = dict(base)
                for model, model_label in zip(MODELS, MODEL_LABELS):
                    source = data[model][target]
                    row = dict(model=model, model_label=model_label, **base, target=target)
                    row.update({out: source[key] for out, key in FIELDS.items()})
                    output.append(row)
                    wide[model] = (f"{float(source['mean']):.4f} "
                                   f"[{float(source['lower']):.4f}, {float(source['upper']):.4f}]")
                    if kind == "within_period" and contrast == CONTRASTS[0]:
                        overview.append(row)
                reading.append(wide)
    return output, reading, overview, inputs


def allocation_values(results, fallback):
    """Read the conditional database, or the permitted aggregate export in the public package."""
    database = results / "current" / "conditional.sqlite"
    rows = []
    if database.exists():
        expected_fits = {(m, i) for m in MODELS[:2] for i in range(64)}
        seen = set()
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
            query = ("SELECT job,replicate,status,details,arrays,sha256 FROM fits "
                     "WHERE family='allocation' ORDER BY job,CAST(replicate AS INTEGER)")
            for job, replicate, status, details, blob, digest in db.execute(query):
                key = job, int(replicate)
                if key in seen or status != "passed" or hashlib.sha256(blob).hexdigest() != digest:
                    raise ValueError(f"Invalid retained allocation: {key}")
                seen.add(key)
                names = json.loads(details)["contrast_names"]
                with np.load(io.BytesIO(blob), allow_pickle=False) as arrays:
                    means, errors = arrays["contrast_mean"], arrays["contrast_sd"]
                if means.shape != (len(names),) or errors.shape != means.shape:
                    raise ValueError(f"Invalid allocation-array dimensions: {key}")
                for target in TARGETS:
                    i = names.index(target)
                    mode, sd = float(means[i]), float(errors[i])
                    rows.append(dict(job=job, replicate=int(replicate), target=target,
                                     mode=mode, sd=sd, lower=mode - NORMAL_975 * sd,
                                     upper=mode + NORMAL_975 * sd))
        if seen != expected_fits:
            raise ValueError("Expected exactly 64 allocations under each linear grid")
        inputs = {"results/current/conditional.sqlite": sha256(database)}
    else:
        rows = read_tsv(fallback)
        rows = [dict(job=r["job"], replicate=int(r["replicate"]), target=r["target"],
                     **{k: float(r[k]) for k in ["mode", "sd", "lower", "upper"]}) for r in rows]
        inputs = {"paper/allocation_fit_values.tsv": sha256(fallback)}
    expected = {(m, i, t) for m in MODELS[:2] for i in range(64) for t in TARGETS}
    if len(rows) != 384 or {(r["job"], r["replicate"], r["target"]) for r in rows} != expected:
        raise ValueError("Incomplete or duplicated allocation contrast family")
    for row in rows:
        if not all(np.isfinite(row[k]) for k in ["mode", "sd", "lower", "upper"]) or row["sd"] < 0:
            raise ValueError("Invalid allocation estimate or standard error")
        if row["lower"] != row["mode"] - NORMAL_975 * row["sd"] or row["upper"] != row["mode"] + NORMAL_975 * row["sd"]:
            raise ValueError("Allocation interval differs from the specified inverse-Hessian interval")
    return rows, inputs


def summarize_allocations(rows):
    summary, display = [], []
    for job in MODELS[:2]:
        for target, label in zip(TARGETS, TARGET_LABELS):
            selected = [r for r in rows if r["job"] == job and r["target"] == target]
            modes = np.array([r["mode"] for r in selected])
            lower = np.array([r["lower"] for r in selected])
            upper = np.array([r["upper"] for r in selected])
            q = np.quantile(modes, [0, .025, .5, .975, 1])
            row = dict(job=job, target=target, n=len(selected), minimum=float(q[0]),
                       q025=float(q[1]), median=float(q[2]), q975=float(q[3]), maximum=float(q[4]),
                       positive_modes=int(sum(modes > 0)), negative_modes=int(sum(modes < 0)),
                       lower_above_zero=int(sum(lower > 0)), upper_below_zero=int(sum(upper < 0)),
                       minimum_conditional_lower=float(lower.min()), maximum_conditional_lower=float(lower.max()),
                       minimum_conditional_upper=float(upper.min()), maximum_conditional_upper=float(upper.max()))
            summary.append(row)
            direction = "positive" if target != TARGETS[2] else "negative"
            count = row["positive_modes" if direction == "positive" else "negative_modes"]
            excluded = row["lower_above_zero" if direction == "positive" else "upper_below_zero"]
            display.append(dict(contrast=label, grid="Original" if job == MODELS[0] else "100 km shifted",
                                median=f"{q[2]:.3f}", empirical_range=f"[{q[1]:.3f}, {q[3]:.3f}]",
                                mode_direction=f"{count}/64 {direction}", interval_exclusion=f"{excluded}/64"))
    return summary, display


def comparison_html(reading):
    esc = html.escape
    head = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Complete common-period comparisons</title><style>body{font:16px/1.55 Georgia,serif;color:#222;max-width:1400px;margin:30px auto;padding:24px}table{border-collapse:collapse;width:100%;font:13px/1.45 Arial,sans-serif}th,td{padding:9px;border:1px solid #ccd4d8;text-align:left}th{background:#edf1f3}td:nth-child(n+3){white-space:nowrap;font-variant-numeric:tabular-nums}.scroll{overflow:auto}h2{margin-top:34px}@media print{thead{display:table-header-group}tr{break-inside:avoid}}</style><h1>Complete common-period comparisons</h1>'
    head += '<p>All three components, seven lineage contrasts, four within-period comparisons and six direct period differences under four national specifications: 210 comparison rows and 840 estimates.</p><p>Entries are posterior means [equal-tail 95% intervals], conditional on the specified exploratory comparisons. Positive values indicate the larger slope for the first-named lineage. Direct differences subtract the second period’s contrast from the first using aligned joint draws. Composite I1 retains the fixed 237:207:95 coefficient weights. The specifications use the same observations and are not independent replications. Intervals do not adjust for the preceding search.</p><p>The full-precision TSV preserves all estimates and Monte Carlo diagnostics. The display is rounded to four decimal places.</p>'
    for component in ["PC1", "PC2", "PC3"]:
        head += f'<h2>Component {component[-1]}</h2><div class="scroll"><table><thead><tr><th>Period or direct difference</th><th>Lineage contrast</th>'
        head += ''.join('<th>' + esc(m) + '</th>' for m in MODEL_LABELS) + '</tr></thead><tbody>'
        for row in reading:
            if row["component"] == component:
                head += '<tr>' + ''.join('<td>' + esc(row[k]) + '</td>' for k in ['period_comparison', 'lineage_contrast', *MODELS]) + '</tr>'
        head += '</tbody></table></div>'
    return head + '</html>\n'


def plot_components(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ["#21618c", "#137b72", "#bc601b", "#767676"]
    markers = ["o", "s", "^", "D"]
    labels = ["Linear · original grid", "Linear · shifted grid", "Quadratic · original grid", "Quadratic · shifted grid"]
    lookup = {(r['component'], r['period_comparison'], r['model']): r for r in rows}
    expected = {(c, p, m) for c in ['PC1', 'PC2', 'PC3'] for p in PERIOD_LABELS for m in MODELS}
    if len(rows) != 48 or set(lookup) != expected:
        raise ValueError("Expected the complete 48-row I1–Z284 overview")
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.6, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'sclara-component-overview',
                         'text.color': '#222222', 'axes.labelcolor': '#222222'})
    fig, axes = plt.subplots(1, 3, figsize=(174 / 25.4, 116 / 25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.19, right=.985, top=.79, bottom=.28, wspace=.14)
    offsets = [.24, .08, -.08, -.24]
    for j, (component, ax) in enumerate(zip(['PC1', 'PC2', 'PC3'], axes)):
        ax.axvline(0, color='#929292', lw=.8)
        for i, model in enumerate(MODELS):
            selected = [lookup[component, p, model] for p in PERIOD_LABELS]
            means = np.array([float(r['posterior_mean']) for r in selected])
            lower = np.array([float(r['lower_95_equal_tail']) for r in selected])
            upper = np.array([float(r['upper_95_equal_tail']) for r in selected])
            if not (np.all(lower >= -.8) and np.all(upper <= 1)):
                raise ValueError("An interval exceeds the displayed common scale")
            ax.errorbar(means, np.arange(3, -1, -1) + offsets[i], xerr=[means - lower, upper - means],
                        fmt=markers[i], color=colors[i], markersize=3.8, elinewidth=1, capsize=1.5)
        ax.set_title(f'Component {j + 1}', fontsize=10, pad=16)
        ax.text(-.015, 1.09, chr(97 + j), transform=ax.transAxes, fontsize=10, weight='bold')
        ax.set_xlim(-.8, 1); ax.set_ylim(-.52, 3.52)
        ax.set_xticks([-.8, -.4, 0, .4, .8]); ax.tick_params(axis='x', labelsize=8.6, length=3)
        ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels(PERIOD_LABELS, fontsize=8.6)
        ax.tick_params(axis='y', length=0, pad=8)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#aaaaaa')
    fig.suptitle('I1–Z284 across all three archaeological components', x=.58, y=.96, fontsize=11)
    fig.text(.58, .88, 'Conditional posterior means and equal-tail 95% intervals', ha='center', fontsize=8.6)
    fig.supxlabel('I1 − R1a-Z284 coefficient difference', x=.58, y=.18, fontsize=9)
    handles = [Line2D([], [], color=c, marker=m, markersize=4, lw=1, label=l)
               for c, m, l in zip(colors, markers, labels)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.58, .025), ncol=2,
               frameon=False, fontsize=8.6, columnspacing=1.8)
    fig.savefig(output / 'Figure_S4.pdf', facecolor='white', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(output / 'Figure_S4.svg', facecolor='white', metadata={'Date': None})
    fig.savefig(output / 'Figure_S4.png', facecolor='white', dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--allocation-values', type=Path, default=Path('paper/allocation_fit_values.tsv'))
    parser.add_argument('--output', type=Path, default=Path('build/supplementary'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    comparisons, reading, overview, inputs = complete_comparisons(args.results)
    allocations, allocation_inputs = allocation_values(args.results, args.allocation_values)
    summary, display = summarize_allocations(allocations)
    for name, rows in [('common_period_comparisons.tsv', comparisons),
                       ('common_period_comparisons_reader.tsv', reading),
                       ('component_overview.tsv', overview), ('allocation_fit_values.tsv', allocations),
                       ('allocation_summary.tsv', summary), ('Table_S24.tsv', display)]:
        write_tsv(args.output / name, rows)
    (args.output / 'common_period_comparisons.html').write_text(comparison_html(reading), encoding='utf-8')
    explanation = ('Online Resource 4: complete common-period coefficient comparisons\n\n'
                   'All 840 retained estimates: three components, seven lineage contrasts, four within-period '
                   'comparisons and six direct differences under four national specifications. The reading table '
                   'contains 210 comparison rows. The full-precision TSV includes posterior means, medians, '
                   'standard deviations, equal-tail 95% limits, probabilities of positive contrasts, R-hat, '
                   'effective sample sizes and Monte Carlo standard errors.\n\n'
                   'Positive values indicate the larger archaeological slope for the first-named lineage. '
                   'Direct differences subtract the second period from the first using aligned joint draws. '
                   'Composite I1 retains fixed 237:207:95 coefficient weights. The four specifications share '
                   'observations. The comparisons are exploratory and intervals do not adjust for the preceding '
                   'search. The HTML and reading TSV use four decimal places; use the full-precision table for '
                   'numerical calculations. These are aggregate results; no new models are fitted.\n')
    with zipfile.ZipFile(args.output / 'ESM_4.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in ['common_period_comparisons.tsv', 'common_period_comparisons_reader.tsv', 'common_period_comparisons.html']:
            archive.write(args.output / name, name)
        archive.writestr('README.txt', explanation.encode('utf-8'))
    plot_components(overview, args.output)
    manifest = dict(comparisons=840, reading_rows=210, plotted_estimates=48, allocation_fits=128,
                    allocation_contrast_values=384, allocation_summary_rows=6,
                    inputs={**inputs, **allocation_inputs},
                    files={p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name != 'manifest.json'})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['inputs', 'files']}, indent=2))


if __name__ == '__main__':
    main()

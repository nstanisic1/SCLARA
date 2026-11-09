# SCLARA

Spatial Concordance Between Paternal Lineages and Stratified Archaeological Records.

SCLARA relates reported paternal origins to dated archaeological landscapes and interprets the associations through source evidence, spatial analysis and comparison with ancient DNA. The Swedish demonstration uses a common archaeological measurement for the Late Neolithic, Bronze Age, Roman Iron Age and Migration period, followed by separate Viking and medieval profiles.

## Public distribution

This distribution contains code, frozen specifications, synthetic examples, tests, aggregate numerical results, the public archaeological-coordinate inventory, and the paper. It excludes participant-linked paternal records, restricted source snapshots, coefficient draws, and the full research workspace.

After installing `requirements.txt`, these commands run with the files supplied here:

```text
python -m unittest discover -s tests -v
python -m examples.minimal
python -m examples.minimal --all-multiple --output build/example_multiple
python -m analysis.supplementary
```

The last command rebuilds Figure S4, Table S24 and Online Resource 4 from the supplied aggregate results; it fits no new model. The complete Swedish reconstruction instructions below describe the full controlled archive and require its authorised inputs. References to retained coefficient draws, databases and `results/reproduction.json` also describe that full archive; those files are not included here. See `DATA_ACCESS.md` and `SOURCES.tsv` for the access and attribution arrangements.

## Run a small example

With the declared Python dependencies installed, run:

```text
python -m examples.minimal
```

This self-contained example creates thirteen synthetic archaeological sites and 120 hypothetical units, including 24 with alternative reported origins. Shared assay dates stay shared across feature assertions. Both periods use the same fixed building-minus-hearth axis. The output in `build/example/` contains source inputs, measured arrays, conditional lineage contrasts and the direct period difference. Everything in this example is synthetic; its intervals are inverse-Hessian approximations. Add `--all-multiple --output build/example_multiple` to give every unit two origins and exercise the no-singleton case.

The reusable input contract is deliberately small:

| Input | Required meaning |
| --- | --- |
| Archaeological assertions | Stable site ID, feature family, material age/error and shared calibration key. Assertions of the same laboratory determination share one date draw. Repeating an assertion does not increase nearest-site exposure. |
| Site and geometry records | Site representative and one or more geometry representatives, coordinate meaning, source uncertainty and position-group identity. Coincident geometries share a positional stream. |
| Calendar distributions | Finite calendar values and their probabilities from the declared calibration or supplied dating model. The small example supplies synthetic dates directly. |
| Periods and feature basis | Ordered half-open bounds, fixed family order, reference centres/scales, loadings and eigenvalues. The measurement function projects through this basis; it does not select a favourable component. |
| Paternal units | One categorical label per reconciled unit and its consecutive locality rows. `counts` states how many equally weighted alternative origins belong to that unit. |
| Contrasts and uncertainty | Explicit coefficient vectors applied jointly. A period difference uses the fitted covariance between periods, not separate significance decisions. |

Within an observed source universe, zero eligible sites gives zero proximity and one eligible site supplies that site's proximity. Unknown coverage is not a zero observation; archaeological support remains a separate qualification. The example, empty/singleton tests and shared-assay checks can be run with `python -m unittest discover -s tests -v`. With the declared R dependencies, `Rscript tests/check_stan_contract.R` compares compiled singleton, all-multiple-origin and mixed likelihoods and gradients with the frozen Python reference values in `tests/stan_contract_inputs.json`.

## Scientific files

- `SOURCES.tsv` identifies the empirical inputs, original providers, versions, source roles and checksums. The public `data/` directory contains `later_position_streams.json`; the other empirical inputs are supplied separately under the applicable access conditions.
- `analysis/` contains source preparation, landscape measurement, the categorical locality-mixture model and frozen Swedish settings.
- `results/` contains the aggregate posterior contrast tables and parameter diagnostics. Complete coefficient draws, conditional families and verification receipts are retained in the full computational archive.
- `paper/` contains the editable manuscript and supplement, their PDFs (the supplement is Online Resource 1), and the principal and supplementary figure masters. Online Resource 2 is `paper/archaeological_coordinates.tsv`; the full participant-linked coordinate inventory remains confidential. `SOURCES.tsv` is Online Resource 3. The DOCX files are the authoritative prose sources. Online Resource 4 supplies all 840 common-period comparisons; `paper/common_period_comparisons.tsv` retains their full-precision values.
- `requirements.txt` and `renv.lock` record the Python and R dependencies.

The complete computational archive, available under the access conditions in `DATA_ACCESS.md`, preserves full file bytes and uses LZMA compression for SQLite entries. Extract it with Python's standard library (`python -m zipfile -e SCLARA_reproducibility.zip .`) or a ZIP reader supporting LZMA. Its `MANIFEST.sha256` lists every scientific file checksum.

In the complete computational archive, source bundles retain third-party genetic, genealogical, archaeological and geographical material under their respective access conditions. Their preservation does not establish collective redistribution permission. Original assertions remain separate from source corrections.

To reacquire an input, use `SOURCES.tsv` and the corresponding catalogue or manifest in the complete computational archive. Retrieve the recorded URL/version and verify the supplied SHA-256 before rebuilding. Report catalogues identify original publications and the pages retained, including cover-only records. The authoritative publication citations and source qualifications are in `SOURCES.tsv`; original acquisition metadata and source bytes remain preserved. A changed live table or newly issued source version is a new input, not a replacement that can silently inherit the frozen analysis. Some third-party snapshots may require direct access from their provider; their presence in this local archive is not a redistribution licence.

## Recompute the Swedish profiles

The checked environment uses Python 3.12.14 and R 4.2.2, with a C++ toolchain for RStan (Rtools42 on Windows). Install Python dependencies in an isolated environment with `python -m pip install -r requirements.txt`. Restore the R package versions and their recursive dependencies with `renv::restore(lockfile='renv.lock', library='build/R-library', prompt=FALSE)`, installing `renv` first if needed. Set the `R_LIBS_USER` environment variable to the absolute path of that restored library before running the sampler. A new R process then uses those locked packages. The Python and R dependency sets have both been restored in fresh local environments.

Run from the repository root. The first two commands regenerate the canonical derivatives from the retained source archives. Measurement and inference outputs are written to `build/`.

```text
python -m analysis.prepare paternal --source data/sources/paternal.zip --output data/paternal.json
python -m analysis.prepare archaeology --source data/sources/archaeology.zip --output data/archaeology.sqlite
python -m analysis.measure --data data --specification analysis/specification.json --output build/measurements --sensitivities
python -m analysis.infer export --measurements build/measurements --specification analysis/specification.json --output build/inference
Rscript analysis/sample.R build/inference
python -m analysis.infer sensitivities --measurements build/measurements --specification analysis/specification.json --output build/inference --workers 2
```

The measurement command includes all fixed source, archaeological-site, neighbouring-pair and paternal-location scenarios. Its `common_transport.json` records the retained within-period variance and proximity-equivalent coverage on the common cohort, with measurement and specification checksums. The export checks conditional modes and derivatives before sampling. The sampler runs eight prespecified posteriors, retaining coefficient draws, contrast summaries and convergence diagnostics; one numerical extension is permitted only when diagnostics fail. The final command reconstructs the complete finite family of 2,628 conditional sensitivity fits and can resume with unchanged inputs.

Common-period coefficients and direct differences use the same frozen archaeological basis. Later profiles have their own meanings and adjustments. Conditional posterior intervals, inverse-Hessian intervals and ranges across sensitivity fits are distinct quantities. Neither a posterior interval excluding zero nor a sequence of coefficient signs establishes population continuity or a historical mechanism.
## Recompute the retained chronology evaluations

The original posthole–hearth comparison uses four earlier windows and its own frozen normalization. Rebuild its source pool and positional measurements, then compare date allocation within the original records with expansion of the archaeological pool:

```text
python -m analysis.chronology prepare
python -m analysis.chronology paired --workers 2
python -m analysis.benchmarks --workers 2
```

These commands use the preceding `build/measurements` output. The paired batch contains 256 allocation fits and two strict-panel checks. The two benchmarks contain 2,000 and 17,400 conditional fits. Every declared condition is retained, including selective recording, dependence, omitted geography and adverse predictive performance. Results resume only with unchanged inputs and code; `--limit` creates an explicitly partial run.

`data/benchmarks/` contains the immutable generating designs and parameters defining these synthetic evaluations. The first fixture predates the final archaeological geometry correction. Preparation independently rebuilds the chronological design and checks its nuisance fit; simulation uses the frozen parameters to preserve its exact generating experiment. The conditions were specified with knowledge of the empirical results. These evaluations assess conditional estimation, temporal attribution and prediction, not calibration of historical profile selection or lineage arrival.

The two prespecified upward branch cuts share the source-unit construction in `analysis/units.py`. Rebuild all four older branch-cut posterior inputs and sample them with the same likelihood:

```text
python -m analysis.branches
Rscript analysis/sample.R build/branches
```

The cut caps of two and five are fixed. Selected records, excluded records, family grouping, alternative origins and reference centring are preserved in the resulting selection inventory. More retained records under a coarser cut do not mean more independent families.

Reproduce the ten retained original-window posterior appraisals—both grids with baseline, covered-cohort, quadratic-geography, joint Kvarnbacken omission and Bobygden omission fits—and the four conditional Kvarnbacken single-site checks:

```text
python -m analysis.appraisal
Rscript analysis/sample.R build/appraisal
```

The original analysis restricted by material quality was not fitted because it did not meet the fixed support criteria. It is not treated as a zero or negative effect.

## Recompute current-profile representation and support checks

The finite extension specification preserves the original feature basis, lineage contrasts and priors. Prepare the cap-2/cap-5 cohorts, fit all 32 conditional designs, sample the eight focal full-cut designs, and complete the fixed crossed-weight families:

```text
python -m analysis.representation prepare
python -m analysis.representation fit-modes
Rscript analysis/sample.R build/representation/posterior
python -m analysis.representation reweight-branches --workers 2
python -m analysis.representation common-support
python -m analysis.representation reweight-support --workers 2
```

Both full-cut and baseline-record companion cohorts are retained. The shared-support comparison subsets whole units by local recorded-site coverage in the Late Neolithic and Roman windows; it does not redefine the archaeological measurement or eliminate environmental confounding. Together these commands produce eight posterior jobs and 3,234 conditional fits. Support decisions, recovered records and excluded units remain explicit.

The current-method evaluation compares the unit-level locality mixture, averaging covariates within units, and weighting by source-record multiplicity:

```text
python -m analysis.evaluation prepare
python -m analysis.evaluation run --workers 2
```

All eight conditions, 1,600 datasets and 4,800 method records are retained. Clean-condition methods are algebraically identical and reuse one fitted solution. Coverage refers to conditional mode/inverse-Hessian intervals. Test labels are independently generated at the same covariates and block effects; predictive scores do not measure geographical extrapolation. This experiment does not calibrate archaeological profile selection, posterior inference or historical conclusions.

## Transfer the framework

`analysis/landscapes.py` and `analysis/model.py` provide the reusable measurement and likelihood interfaces. The source preparation and frozen settings in this repository are Swedish adapters. A new application should make its own source definitions explicit:

1. Define the historical alternatives, geographical meaning of an origin and branch nomenclature. Reconcile assertions before forming units; preserve alternative localities and record why any unit is withheld.
2. Separate archaeological feature identity, material date and activity phase. Keep source corrections attached to original assertions. Supply dated site geometries and calendar distributions with shared assay identities.
3. Define period windows, feature families, distance kernel and reference population. Inspect source support before interpreting a profile. Freeze its centring, scales, loadings and orientation before comparing periods.
4. Fit one categorical observation per unit, averaging locality probabilities inside the likelihood. Preserve joint covariance when deriving lineage and period contrasts. Distinguish conditional posterior intervals from measurement and dependence sensitivities.
5. Compare the spatial results with independently dated ancient evidence and original archaeological contexts. Judge a profile by construct validity, support, historical discrimination and sensitivity together. An informative unresolved contrast can be retained with its uncertainty; it does not establish equivalence or continuity.
6. Retain the full declared result family, adverse findings, source exclusions and checks. Test recovery, attribution and prediction under the actual estimator and observation process rather than assuming that one successful benchmark validates every application.

## Reconstruct the ancient evidence

```text
python -m analysis.ancient
```

`data/ancient/selection.json` records the finite source-row selection, documented aliases and retained crosswalk checks. `sources.json` identifies and hashes all sixteen original files. The command reads those sources and rebuilds the verified assertions, individual identities, paternal branch crosswalks, published date variants and relatedness assertions in `build/ancient/`. It also produces `eligibility.json`, an inventory for each question that records source-specific quality flags, population-analysis exclusions, date conflicts, records retained only as metadata, and reconciled branch assignments for the individuals shown in the figures. These classifications use no archaeological landscape outcomes. Source rows shared across questions do not become additional people. Published relatives remain separate individuals, and broad paternal assignments are not promoted to unsupported downstream branches.

## Inspect retained results and rebuild figures

The complete computational archive contains the 22 original diagnosed posterior fits (92,000 draws), eight current-profile branch posterior fits, 2,628 original and 3,234 extension conditional fits, 258 paired chronology fits, 19,400 inherited benchmark fits and 4,800 current-method evaluation records. Four Kvarnbacken single-site conditional checks and the original support decisions are also retained. Every sampled coefficient, chain identity and iteration is preserved. Lossless byte-shuffled float64 NPZ storage replaces CSV draw archives; the original CSV hashes and exact numerical-value hashes remain in the sampling receipts. No draws are thinned or rounded. Contrast draws are reconstructed from those coefficients and the frozen contrast vectors, avoiding duplicate columns. The stored conditional-model outputs are preserved; the older database summaries now use the common file format. These archived calculations remain distinct from a fresh inference run.

After generating the design exports above, reconstruct every retained contrast and summary from the coefficients and verify the finite result families:

```text
python -m analysis.results --verify
python -m analysis.extensions verify
```

Build the principal figures from the retained results and the regenerated measurements and ancient evidence:

```text
python -m analysis.figure_data
python -m analysis.figures --input build/figure_data --output build/figures
python -m analysis.coordinates
```

`analysis.results` checks draw ordering, numerical reconstruction, file integrity and the saved sampling diagnostics; it does not replace rerunning the sampler. `analysis.figure_data` retains the complete declared common and later contrast sets and adds no model fit or profile search.

`data/reports/reviews.json` holds the eighteen clearance-context reviews and seven later-context reviews. The accompanying 27 PDFs preserve 247 original whole pages, including report identification and every retained review/correction page locator. `data/reports/sources.json` maps the excerpt page numbers to original PDF page numbers, identifies the full original files and records source-specific access conditions. Page references in the reviews refer to the original PDFs. Four further unchanged original reports, including Kvarnbacken, are members of the archaeological source archive.

## Rebuild the supplementary comparisons

```text
python -m analysis.supplementary
```

This command rebuilds Figure S4, Table S24 and Online Resource 4 (`ESM_4.zip`) in `build/supplementary/`. It retains all 840 estimates across the three components, seven lineage contrasts, four periods and six direct period differences under four specifications. The 384 allocation contrast values summarize 128 retained conditional fits. The full archive reads these fits directly from the conditional database; the public distribution uses the identical aggregate export in `paper/allocation_fit_values.tsv`. No model is fitted. The generated manifest records input and output checksums. `paper/common_period_comparisons.tsv` retains the full-precision comparison table.

After collecting a complete fresh inference run, use `python -m analysis.supplementary --results build/results --output build/fresh_supplementary` to make the same displays from its results.

## Reconstruct the northern candidate

```text
python -m analysis.northern
```

The command reads the original Hennius workbook, two Rimpi tables and a source-defined subset of the national site register. Its 249 native register objects include every matching legacy-locator candidate, explicit system identifier and exact laboratory-linked site; ambiguous alternatives and original geometry/uncertainty remain present. It independently reconstructs the 439 assertions, laboratory and site crosswalks, inherited phase census, branch-support counts and northern posthole/hearth totals. It fits no lineage effect. `data/northern/appraisal.json` gives the retained historical interpretation, and `results/northern.json` preserves the checked result.

`results/historical_screen.sqlite` retains the complete earlier archaeological screen: 1,664 axis representations across thirteen panels and 74 features, all 9,984 contrasts, 1,352 fit diagnostics, twelve nominated axes and 48 recorded sensitivities. Its source qualifications and historical questions preserve the earlier candidate assessment. These are outcome-informed development results preceding the current common-period and corrected later analyses. They do not provide independent confirmation of those analyses; all retained Holm-adjusted values equal one.

## Collect a complete fresh run

After all preceding model, chronology, benchmark and northern commands finish, collect their outputs in a new directory. This validates the complete families and sampling receipts without replacing the retained snapshots:

```text
python -m analysis.collect
python -m analysis.results --results build/results --output build/fresh_results_summary.json.gz --verify
python -m analysis.extensions verify --results build/results --output build/fresh_extensions
python -m analysis.figure_data --results build/results --output build/fresh_figure_data
python -m analysis.figures --input build/fresh_figure_data --output build/fresh_figures
```

The collection requires all 30 posterior jobs, 5,862 conditional fits, 258 paired chronology fits, 19,400 inherited benchmark fits and 4,800 current-method evaluation records. It also includes the appraisal checks and reconstructed northern inventory. The earlier historical screen remains an archived development record. Repeated posterior sampling can differ within Monte Carlo uncertainty; assess the means and interval endpoints with their retained Monte Carlo standard errors and check convergence again.

`python -m analysis.coordinates` rebuilds both `build/paper/coordinates.tsv` and `build/paper/archaeological_coordinates.tsv` from canonical records, generated common-period and later measurements, original-context reviews and assembled ancient evidence. The public archaeological inventory (Online Resource 2) contains 9,045 entries with site names, source locators, decimal and DMS coordinates, geographical meaning and reliability. The full inventory adds 1,404 participant-linked paternal-origin positions and is confidential reviewer material; it is excluded from public supplementary files and the public code distribution. The manuscript and supplement can be edited directly in Word-compatible software; their final numerical tables refer to the retained result files.

## Verification of the complete computational archive

The current archive is version 2026.11.09.4. The complete research archive retains numerical verification under version 2026.11.08.3, with subsequent editorial and reproducibility checks recorded separately.

`results/reproduction.json` distinguishes the complete earlier replay from the finite extension checks. The earlier reconstruction reran all 22 original posterior jobs, 2,628 conditional fits, 258 paired fits and 19,400 benchmark fits in freshly restored environments. It preserves the declared numerical tolerances and nine threshold-sensitive calibration-bin exceptions; other discrete benchmark decisions match.

For the revision, canonical scientific fields, all 17 measurement files and all design exports reproduce exactly. Independent reruns cover all 36 extension modes, 24 prospectively selected crossed-weight fits across every design, 24 regenerated benchmark datasets (72 method records across every condition), and two prospectively selected common/later posterior jobs. Every coefficient, contrast and diagnostic from all eight extension posteriors, all 3,234 extension conditional fits and all 4,800 evaluation records receives full retained-result verification. This is distinct from resampling all eight extension jobs. Interval-exclusion changes between retained and rerun posteriors are reported explicitly.

The original retained draws remain the paper estimates. Lossless storage changes preserve every float64 coefficient and its chain/iteration identity. The full calculation takes several hours; the synthetic example provides a short first run.

Release 2026.11.08.3 preserves later-period positional draws through a versioned registry of stable archaeological position IDs, canonical centres, radii and original seeds. Seven explicit, source-qualified ancient aliases yield 275 identity groups and 44 links while preserving all 149 mapped observations. The coordinate inventory corrects fifteen display labels, and the document PDFs retain the original figure resolution. All affected measurements, conditional contrasts and figure masters reproduce exactly; the retained posterior draws are unchanged. Earlier verification records retain their original dates and scope.

## Data access and citation

Repository: [https://github.com/nstanisic1/SCLARA](https://github.com/nstanisic1/SCLARA). Software release: [10.5281/zenodo.24088352](https://doi.org/10.5281/zenodo.24088352) (version 2026.11.09.4).

Stanisic, N. (2026). *SCLARA: Spatial Concordance Between Paternal Lineages and Stratified Archaeological Records* (version 2026.11.09.4) [Software]. Zenodo. [https://doi.org/10.5281/zenodo.24088352](https://doi.org/10.5281/zenodo.24088352).

`SOURCES.tsv` provides the file-level data citations and source locators. `CITATION.cff` identifies the software author and release. `DATA_ACCESS.md` distinguishes the complete local research archive from the public code distribution and explains the access conditions for empirical inputs.

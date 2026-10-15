# Proposed study: graph structure and the propagation of scientific activity

Research design date: 2026-10-14. Status: proposal informed by exploratory diagnostics. Outcome definitions and model choices require freezing before new evaluation.

## 1. Research question and scope

Do scientific relationship graphs explain and predict the propagation of departures from expected research activity? Can the resulting forecasts identify sustained adoption across research communities before a temporal baseline does?

The primary contribution would be evidence about when graph geometry constrains scientific dynamics. Graph Fourier analysis is one candidate representation. A useful result could also identify where its assumptions fail and where a directed or learned temporal operator performs better.

Publication activity, concept recombination, adoption breadth, and scientific value are different outcomes. The first study should measure activity and adoption. Claims about breakthroughs or scientific quality require separate evidence.

## 2. What the Fourier intuition implies mathematically

For a symmetric graph Laplacian `L = U Lambda U^T`, graph Fourier coefficients are `alpha_t = U^T x_t`. The map is orthogonal and invertible. With the same observations and an unrestricted estimator, a fixed invertible change of coordinates preserves information about the target. Improvements arise through denoising, regularization, lower-dimensional estimation, or a better approximation to the evolution law.

In physics, Fourier modes are useful when the governing operator is simple in that basis. The corresponding hypothesis here is that a scientific relationship graph approximately organizes the dynamics of research activity. A Laplacian measures geometry; it supplies no empirical law of scientific evolution on its own.

| Representation | Object decomposed | Possible benefit | Necessary empirical check |
| --- | --- | --- | --- |
| Graph Fourier | Laplacian of scientific relationships | Shared estimation of dynamics over related areas | Graph alignment improves held-out prediction and survives graph nulls |
| PCA or dynamic factor model | Covariance of observed activity | Removes common trends and reduces dimension | Factor structure remains useful in later periods |
| DMD or an estimated linear transition | A fitted map between successive states | Identifies patterns with estimated growth, decay, or oscillation | Transition estimates remain stable and forecast later observations |
| Graph wavelets | Localized functions of the graph Laplacian | Resolves disturbances by neighborhood and scale | Local features improve on the global and ordinary local-network baselines |

Graph frequency measures differences over connected nodes. Temporal frequency measures variation over time. High graph frequency alone provides no direct evidence of early emergence. A symmetric graph also contains no direction of transfer. Any interpretation as directional spread needs additional temporal evidence or a directed relationship model.

The original implementation transforms a recent three-year difference over the graph and maps the filtered result back to nodes. It does not fit a law for how graph modes evolve in time. The proposed study adds that missing predictive step.

## 3. Diagnostics of the existing experiment

The [local script](audit_current_design.py) uses the committed June 2026 data, the same 2022 cutoff, and the same 250-work eligibility threshold. It writes new evidence under this research directory and leaves the published report intact.

### Filter specificity

A node filter has the form `H = U g(Lambda) U^T`. Its best scalar identity approximation is `c I`, where `c = trace(H)/n`. The diagnostic reports `||c I||_F^2 / ||H||_F^2`.

The existing local filter gives a fraction of 0.9382 and the midband filter gives 0.9916. The latter passes nearly all modes with similar positive weights. This statistic describes the operator; it does not measure explained variance in the observed series. It motivates testing a scalar identity filter and randomized graphs as explicit comparators.

### Outcome decomposition

Let `x_t` be topic CLR at the cutoff. The existing outcome decomposes exactly:

```text
mean(x_(t+1:t+3)) - mean(x_(t-2:t))
  = [mean(x_(t+1:t+3)) - x_t]
  + [x_t - mean(x_(t-2:t))].
```

The first component is entirely subsequent movement. The second is known at the cutoff. For a linear trend, the known component is one third of the full shift. The earlier outcome is valid for sustained relative change, and it can reward continuation of an already established trend. Its correlation therefore needs this interpretation in a forecasting argument. This is target construction, rather than evidence that the feature computation used future data.

| Feature | Original sustained-shift rho | Entirely subsequent-movement rho |
| --- | ---: | ---: |
| Fixed spectral composite | 0.5173 | 0.2359 |
| Midband spectral momentum | 0.5136 | 0.1555 |
| Standardized raw momentum | 0.4800 | 0.1165 |
| CLR growth baseline | 0.4751 | 0.0895 |

These values use 74 topics. The observed component alone correlates 0.5036 with the full shift and 0.0844 with subsequent movement. Across the existing 18 cutoffs, mean correlation with subsequent movement is 0.5980 for the growth baseline and 0.3992 for the fixed spectral composite.

The future-only spectral association remains worth investigating. Its target was chosen after inspecting this dataset; no confirmatory significance claim follows. All previously examined cutoffs and endpoints are development data for the proposed research.

## 4. Hypotheses with explicit failure conditions

### H1. Scientific relationships help predict future residual activity

After conditioning on each area's own history and common corpus-wide factors, a graph estimated from earlier scientific relationships predicts subsequent deviations better than matched temporal models.

Use graph edges derived from pre-cutoff references or concept incidence, with the provenance independent of the activity-correlation calculation. Shared paper sources can still induce dependence; this construction does not establish causality.

The primary comparison is forward predictive loss. H1 fails if graph terms provide no stable improvement over the best temporal/factor baseline, or if matched randomized graphs perform equally well.

### H2. Sustained recombination has precursors beyond popularity

For a pair with weak existing evidence of joint use, changes in the pair's local scientific neighborhoods predict later repeated joint use across independent author groups. Condition on both concepts' frequencies, their growth rates, pair age, existing pair count, common neighbors, and semantic similarity.

This measures the transition from tentative joint use to sustained adoption. It is adjacent to Science4Cast link prediction and recent conceptual-diffusion work. The proposed distinction is the joint temporal test of graph dynamics, repeated use, and adoption outside the originating group.

H2 fails if gains disappear after popularity and local graph controls, if they occur only for already common pairs, or if a temporal burst detector alerts at the same time with comparable precision.

### H3. The useful representation depends on the process

An undirected scientific graph may describe some stable propagation patterns. Common external shocks or directional transfer may favor factor models or a learned temporal transition instead. Estimate this dependence through simulated processes and later time blocks.

A graph Fourier model should reduce residual cross-mode lag covariance and improve forecast loss relative to matched bases. Neither covariance diagonalization nor a visually concentrated spectrum is sufficient on its own. Graph construction and this diagnostic must use separate training and evaluation observations.

H3 is a mechanism comparison. It remains exploratory until the definitions of regimes and diagnostic thresholds are fixed.

### Deferred extension: higher-order combinations

Papers containing three or more concepts retain information lost in pairwise projection. A paper-concept incidence matrix or hypergraph could predict repeated joint use of concept sets. This requires paper-level data and a comparison against pairwise features of equal capacity. It follows H1/H2 and should proceed only if paper coverage and basic predictive dependence are established.

## 5. First model and comparisons

Fit a temporal count model using only information available at each forecast origin. It should include topic intercepts, recent level and growth, seasonality for monthly data, and shared corpus factors. Use a negative-binomial observation model if overdispersion warrants it. Forecast the future corpus exposure separately when using an offset; observed future corpus size belongs only to an explicitly conditional analysis.

Let `r_t` denote standardized one-step residuals from forecasts constructed with past data. Let `S` denote a fixed normalized scientific adjacency within an outer fold. Start with:

```text
r_(t+1) = a0 r_t + a1 S r_t + a2 S^2 r_t + epsilon_(t+1).
```

The model shares a small number of parameters across the graph. For symmetric `S = U diag(s) U^T`, each graph mode evolves with gain `a0 + a1 s + a2 s^2`. That is the direct test of whether a graph basis helps estimate the evolution law. Fit a short autoregressive extension only if the sample size and training validation support it. Residual standardization and all count-model parameters are fitted using historical observations.

The common-trend model should be estimated before claiming cross-area propagation. A shared funding shock, indexing change, or widespread terminology change can generate correlated activity without transmission over the scientific graph.

Use the following comparisons with tuning budgets matched where applicable:

| Comparison | What it tests |
| --- | --- |
| Persistence, local linear trend, independent count autoregression | Whether graph terms add to ordinary extrapolation |
| Burst/change-point detector | Whether an alert is earlier at the same false-alarm rate |
| Dynamic factor/PCA temporal model | Whether shared shocks explain the apparent graph effect |
| Ridge or reduced-rank VAR | Whether a scientific graph is preferable to generic shared temporal estimation |
| Scalar identity and fixed original filters | Whether graph-specific mixing matters |
| Common-neighbor, Adamic-Adar, degree-product and pair-count models | Whether proposed recombination scores add to existing local link predictors |
| Random basis and shuffled graph with matched parameter counts | Whether apparent gains come from regularization alone |
| Small learned temporal operator or DMD | Whether geometric modes align with the useful evolution modes |

An isotropic ridge regression rotated through a full orthonormal basis has equivalent predictions when all preprocessing and penalties are transformed consistently. That equivalence is a diagnostic for an unfair basis comparison. Spectral truncation or mode-specific shrinkage changes the estimator and must be compared with equally constrained PCA and random-basis alternatives.

## 6. Data acquisition in stages

### Existing benchmark first

Use a documented Science4Cast specification to establish a reproducible edge forecast and local graph baselines. Freeze the archive and its checksum, verify edge timestamps, restrict the eligible node universe to concepts observed by each cutoff, and preserve the benchmark's natural positive prevalence for evaluation.

This can test link prediction and spectral features on a known task. Claims about unique paper counts, adoption groups, or citation-mediated transfer require additional fields. Co-occurrence edge multiplicity cannot automatically substitute for paper counts because a paper with many concepts creates many edges.

### Bounded paper corpus for the main scientific question

The proposed first domain is robotics with neighboring machine-learning and computer-vision literature. It offers a plausible setting for method transfer and a subject-defined sampling rule. Use a declared robotics core and predeclared neighboring categories; record which papers belong to each. Outcomes across communities require observing the receiving communities, not just the core corpus. A quantum-information corpus is a possible independent replication after accounting for the directly overlapping 2026 preprint.

Begin with a source and coverage audit before choosing a concept count or model dimension. Monthly aggregation over complete years provides more observations than the present annual panel, although serial correlation limits the effective sample size. Choose monthly versus quarterly resolution using the observed sparsity in training data.

Required fields are paper ID, first public date, text/version used, concept incidence, outgoing references, and authorship provenance. Record missingness by year and community. Deduplicate preprint and journal manifestations with explicit rules; retain unlinked records and uncertainty flags. Treat authors as imperfectly resolved entities. Define an independent adoption group using a frozen collaboration graph or an auditable shared-author rule, and report sensitivity to that definition.

### Broader relationships later

Use OpenAlex or a targeted SciSciNet extract for citation relationships, institutions, funding, or patents if the pilot justifies the added effort. SciSciNet-v2 depends on OpenAlex and provides a different linkage layer rather than independent source validation. Citation impact, patent uptake, and funding are later outcomes or external covariates with their own availability dates.

## 7. Outcomes and eligibility

For H1, the primary target is a future monthly or quarterly count distribution, scored with a proper predictive loss. Report relative-attention change separately using log ratios or another declared compositional model. An area can gain absolute activity while losing corpus share; both quantities are informative.

For H2, define an eligible pair using only evidence available at the origin. Candidate proposals to freeze after the coverage audit are:

- Both concepts have appeared before the cutoff and have adequate training support.
- Joint use is absent or limited at the cutoff; evaluate these two risk sets separately.
- The event is a minimum number of unique joint-use papers over the next 24 months, spread across at least two future time bins and at least three independent author groups.

The event's minimum paper count is a design parameter to set from training-only prevalence and precision requirements. The proposed 24-month horizon and group threshold also require freezing before evaluation. Report a fixed small sensitivity set once the primary rule is frozen. If group provenance is unreliable, use a repeated-publication endpoint and describe its narrower scope.

Separate new combinations of established concepts from previously unseen concepts. A fixed graph has no node on which to forecast a new concept before its appearance. Open-vocabulary detection is a separate problem.

## 8. Temporal validity and uncertainty

Every outer origin freezes corpus membership, vocabulary, concept extraction, graph, scalers, factor model, and fitted parameters. Construct training residuals and graph histories without later observations. Choose hyperparameters only on earlier outcomes whose complete horizons finish before the outer origin. The protocol should include a gap when shared outcome windows would otherwise cross a split.

An old publication date in a current export does not guarantee that all metadata were available historically. arXiv OAI exposes the latest metadata and modification datestamps. Use dated article versions or label the experiment as retrospective reconstruction. Learned text embeddings and LLM concept extraction can import later knowledge through pretraining; a chronological text-only vocabulary and representation provide the strict initial comparison.

The earlier project has examined 2023-2025 outcomes and June 2026 data. Those years cannot be called an untouched test for this research. Reserve a new corpus/time block before inspecting labels, and record all exposure. Add a prospective prediction ledger after the design is frozen; complete outcomes will become evaluable later. Historical backtests, external replication, and prospective evidence should be reported separately.

For uncertainty, use paired forecast-loss differences aggregated by future time block. Resample blocks long enough for the forecast horizon and inspect sensitivity to block length. Network and repeated-pair dependence require additional group-aware checks. If only a few complete event blocks are available, report their individual results and avoid precise asymptotic claims. Avoid treating node rows as independent replicates.

Use average precision, precision at a fixed review budget, calibration, and time-dependent detection rate for rare events. Report the natural base rate beside event metrics. Measure lead time among correctly detected events together with false alarms per active concept-year. A model that alerts constantly can have long apparent lead time without useful discrimination.

## 9. Controls that could falsify the mechanism

| Control | Structure retained | Interpretation |
| --- | --- | --- |
| Graph label permutation, stratified by training activity and age | Topology and the broad activity distribution | Tests whether assigning specific areas to graph positions matters |
| Degree/strength-constrained rewiring | Connectivity or weight heterogeneity to the extent verified | Tests whether the exact scientific neighborhoods matter beyond popularity |
| Independent time surrogates after common-factor removal | Approximate univariate histories and shared factors, under the stated surrogate model | Tests whether residual cross-area timing matters |
| Paper-concept incidence randomization | Training paper sizes and concept frequencies where the sampling method preserves them | Tests whether co-occurrence structure adds beyond marginal frequencies |
| Alternate extraction and corpus rules | The intended scientific population | Tests sensitivity to labels, coverage, and terminology |
| Univariate and shared-shock simulations | Known mechanisms without graph propagation | Measures false discoveries in conditions where graph structure should provide no advantage |

Each null needs its own preservation audit. Permutation significance under an exchangeable-label model alone cannot establish that the graph represents scientific propagation. Predictive association supports a forecasting mechanism; causal transfer requires a separate identification strategy.

## 10. Work sequence and decisions

1. **Completed: design diagnosis and literature scoping.** Reproduced the original holdout comparisons, quantified filter specificity, and separated the outcome's known and future components.
2. **Next: synthetic falsification.** Simulate independent trends, shared shocks, and graph-polynomial propagation at noise and sample sizes resembling the pilot. Include a changing graph and directed transfer as stress tests. Confirm expected successes and failures before fitting the paper corpus.
3. **Then: existing temporal benchmark.** Reproduce a local graph baseline and evaluate a compact spectral model with a strict time split. Stop adding spectral complexity if it cannot improve over local features and null graphs.
4. **Then: bounded paper-level corpus.** Audit text dates, concept incidence, graph coverage, and event prevalence. Freeze a primary question, one outcome, model set, and validation schedule.
5. **Then: external or prospective confirmation.** Evaluate a reserved domain or time block once. Record predictions for a future complete horizon after registering the design.

Proceed toward a forecasting claim only if the method improves proper predictive loss or a predeclared event metric over the strongest comparison, survives graph nulls, and repeats across independent time blocks. Practical improvement thresholds should be set before evaluation using the intended alert-review budget. A single favorable correlation is insufficient.

If common factors explain the gains, study shared shocks and uptake timing. If directed temporal models work and Laplacian models fail, study asymmetric transfer. If graph-local signals improve descriptions without forward gains, write about the limits of spectral emergence indicators. These outcomes can support substantive findings when the research question and evaluation are clear.

## 11. Candidate contribution and article direction

A defensible provisional article topic is **When does scientific network structure help forecast research activity?** A positive result could identify a specific predictive scale and propagation pattern. A negative result could show why spectral emergence scores largely reproduce trends and identify the conditions under which that occurs.

The literature review establishes overlap and motivates these tests. It does not establish priority, and the diagnostic results provide no evidence yet for H1 or H2. The current public article remains the record of the earlier aggregate experiment.

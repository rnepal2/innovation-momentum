# Traffic-wave amplification: focused literature check

Review date: 2026-10-15. This note follows the [congestion-precursor audit](traffic-novelty-review.md) and evaluates a narrower question: can past-only graph-mode spectra predict whether a traffic disturbance will amplify or decay, after accounting for its current conditions?

## Verdict

Proceed with a bounded comparative study. The accessed sources did not establish an exact duplicate of that complete design. Existing work already covers spectral identification of traffic oscillations, amplitude growth, model-based stability, graph Fourier congestion warnings, and dynamical-mode forecasting. The contribution would depend on a demonstrated benefit or explanation specific to the proposed comparison. Universal priority is not established by this search.

A reproducible extension or replication with clear limits remains a useful outcome. Additional figures can explain the mechanisms and comparisons. Their scientific interpretation must follow the measured result.

## Closest work

| Source | Evidence inspected | Overlap and boundary |
| --- | --- | --- |
| [Li, Wang and Ouyang, 2012, Prediction and field validation of traffic oscillation propagation under nonlinear car-following laws](https://doi.org/10.1016/j.trb.2011.11.003); [author institution record](https://experts.illinois.edu/en/publications/prediction-and-field-validation-of-traffic-oscillation-propagatio/) | Publisher abstract and institutional publication record; full methods unavailable in this check. | Uses describing-function analysis and frequency-domain calibration to predict dominant frequency and amplitude growth under nonlinear car-following laws, compared with field trajectories and simulation. Frequency-based amplification prediction is established. This is a model-based vehicle-following approach; an exact GFT-band comparison was not established. |
| [Zhao, Nie and Zhang, 2014, Extended spectral envelope method for detecting and analyzing traffic oscillations](https://doi.org/10.1016/j.trb.2013.12.007) | Publisher-indexed abstract, introduction and section excerpts. Full article access was blocked. | Uses cross power spectral density across locations, short/long windows, and a location contribution measure to characterize evolving oscillations. Simulated examples include topology-dependent traffic oscillations. This is a close spectral baseline. Detailed prospective validation and code were not verified. |
| [Belletti et al., 2015, Prediction of traffic convective instability with spectral analysis of the Aw-Rascle-Zhang model](https://doi.org/10.1016/j.physleta.2015.05.019); [author PDF](https://bayen.berkeley.edu/sites/default/files/prediction_of_traffic.pdf) | Targeted full-text reading: theoretical interpretation, transfer functions, Sections 3.4, 4 and 5; relevant complete pages and Figures 7, 12 and 13 visually inspected. Proofs were not independently checked. | Linearizes a traffic PDE and studies frequency-dependent propagation, gain, damping and model-specific stability. Compares its solution with NGSIM data on a homogeneous freeway segment. Strong physical overlap; its spectral operator and fitted reconstruction differ from the proposed graph-band forecast. Model-specific conclusions cannot be generalized to all real traffic. |
| [Avila and Mezic, 2020, Data-driven analysis and forecasting of highway traffic dynamics](https://www.nature.com/articles/s41467-020-15582-5) | Full publisher main-text methods, growing/decaying-mode discussion, and moving-horizon forecasting section. Supplementary material and code were not independently audited. | Hankel-DMD/Koopman modes identify growing, decaying and persistent patterns on NGSIM data. A moving-window procedure also forecasts PeMS conditions. This directly overlaps the explanatory and prediction goals. It estimates an evolution operator; the proposed GFT uses a specified spatial graph. No matched-state head-to-head test of the proposed graph-band statistic was identified in the inspected sections. |
| [Ling et al., arXiv:1902.11289v1, 2019; journal publication 2022, Koopman Operator Applications in Signalized Traffic Systems](https://arxiv.org/html/1902.11289v1); [author publication page](https://www.coogan.ece.gatech.edu/papers/ling2019koopman.html) | Full-text Section V, Algorithm 3, data/window definitions and surrounding applications. | Rolling DMD estimates queue dynamics; persistent eigenvalue magnitude above one flags instability. The demonstration compares accident-related and ordinary peak-hour queue growth. The input window uses past observations. This settles that past-window spectral growth alerts already exist. The demonstration concerns signalized queues and a learned temporal operator. It does not establish the proposed freeway GFT-band advantage. |
| [Takada, Azuma and Banno, 2025, Prediction of Traffic Congestion Using Graph Fourier Transform](https://doi.org/10.11509/sci.SCI25.0_1017) | All four pages and selected author scripts inspected in the preceding audit. | Direct non-neural GFT precursor prior art: low-mode coefficient dominance warns before a downstream congestion threshold in simulation. The target is event occurrence. The proposed study instead measures subsequent disturbance amplification and tests an explicit graph contribution against strong spectral baselines. |
| [Loukas and Perraudin, arXiv:1611.00255v3, 2019, Stationary time-vertex signal processing](https://arxiv.org/html/1611.00255v3) | Joint stationarity/JPSD definitions and spectral interpretation in Section 3; methodological foundation. | Joint graph-temporal spectral density is established. A graph transform followed by temporal spectra and band integration supplies a measurement, with no intrinsic claim of novelty. Local stationarity and cross-mode dependence require examination. |

Additional screening covered [Raju et al., 2022](https://doi.org/10.1061/AJRUA6.0001227), whose institutional abstract describes DFT-based empirical string-stability assessment, and the incident-analysis ESE preprints [1507.00224](https://arxiv.org/abs/1507.00224) and [1507.00804](https://arxiv.org/abs/1507.00804). Their full texts were not used to establish a gap.

An indexed 2026 OpenReview manuscript, [b6Py2zy0fK](https://openreview.net/forum?id=b6Py2zy0fK), mentions graph-time Fourier neural operators and frequency-band processing. The full manuscript and metadata could not be retrieved through its public page/API during this check. It remains a reading gap; no detailed claim about its results is made here.

## Design consequences

These are my deductions for our proposed experiment:

1. Define the outcome precisely. Amplification along a vehicle platoon, temporal growth at a fixed location, and growth of a moving wave packet are different measurements. Track packet amplitude over its propagation for the primary traffic outcome. A packet entering a fixed observation region can raise local energy without any increase in its own amplitude.
2. Separate disturbance strength from damping. Greater spectral power may reflect stronger forcing. Controlled simulations should vary forcing and response independently; any inferred physical decay rate needs a corresponding response or temporal-dynamics measurement.
3. Check the role of graph geometry. A regular corridor's Laplacian eigenmodes closely relate to ordinary sine/cosine analysis. Include ordinary spatial Fourier/cosine transforms, and test lane/ramp or irregular network structure if claiming an advantage from graph geometry. Such a structure does not guarantee an advantage.
4. Estimate temporal behavior separately. For a local discrete model `x[t+1] = A x[t] + e[t]`, GFT coordinates satisfy `z[t+1] = (U^T A U) z[t] + U^T e[t]`. Laplacian eigenvalues alone do not determine growth. Off-diagonal terms describe coupling between graph modes and should be checked before using independent-mode models.
5. Preserve a prospective test. Use past-only feature windows, training-selected bands, and separate development/test days or simulation runs. Define future amplification through an independent physical amplitude measurement. Control for current speed, density, variance and spatial gradients, with common information and comparable fitting budgets across methods.

## Proposed contribution

Working question:

> Under which sensing and road-structure conditions do graph-mode spectral bands improve prediction of traffic-wave amplification beyond ordinary spatial statistics, spectral envelopes and learned dynamical modes?

The first experiment should distinguish two hypotheses:

- Predictive hypothesis: graph-band measurements add held-out information about future amplification after current-condition controls.
- Representation hypothesis: the specified road graph provides an advantage over matched alternative bases. This advantage may depend on irregular geometry or limited observations.

A positive control can recover known dynamics; it validates the analysis. Empirical contribution requires an out-of-sample result or a well-supported limit. A result shared by earlier methods should be described as replication or corroboration.

The necessary comparisons are ordinary Fourier/cosine analysis, PCA, cross-spectral envelope analysis, Hankel-DMD/Koopman methods, basic traffic-state predictors and randomized graph/basis controls. Test whether any advantage survives changed sampling intervals and sensor availability. Formalization of the spectral-envelope baseline needs its complete method before implementation.

## Search and provenance

Targeted queries combined graph Fourier, graph bands, PSD/CPSD, amplification/decay, traffic string stability, convective instability, spectral envelope and Koopman/DMD. Citation trails linked the envelope work to earlier frequency-domain vehicle-following studies. This was a quick scoping review, with explicit reading depths above, and no exhaustive absence claim.

The author-hosted Belletti PDF is stored in the ignored directory `traffic/data/literature/`. SHA-256: `9060310cc4e79fc64db7c0525c2f64c55944009e17fff5e45db2b702f71712e3`. No manuscript or rendered figure is added to Git.

No traffic data were downloaded, no traffic experiments were run, and the public article and analysis code were left unchanged.

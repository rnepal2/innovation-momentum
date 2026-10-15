# Research directions: scientific activity on graphs

Research review date: 2026-10-14. Branch: `feature/research-directions-2026`.

## Experiment Results

The [completed robotics campaign](campaign/findings.md) tests both hypotheses with OpenAlex and arXiv records, simulations, graph nulls and chronological evaluation. H1 has no graph-specific forecasting support in this pilot. H2 has small fixed-vocabulary gains with mixed vocabulary/source sensitivities. The [experiment directory](campaign/README.md) contains reproduction commands, outputs and the decision ledger. The proposal below records the starting design.

The [cross-community adoption follow-up](adoption/findings.md) tests citation/collaborator exposure, community breadth and directed citation paths on held-out medical-imaging periods. None meets the practical improvement gate. The [reproduction directory](adoption/README.md) includes the executed notebook, prospective ledger and measurement limitations.

The [traffic study](traffic/findings.md), on `feature/traffic-spectral-amplification`, includes numerical controls, a public-data audit, SUMO pulse responses, and a phase-aware forecasting follow-up. It establishes no empirical GFT forecasting advantage. The [traffic directory](traffic/README.md) contains protocols, results, and reproduction commands.

## Recommendation

Study whether independently measured scientific relationships help predict how unexpected research activity spreads between areas. Test the conditions under which graph spectral coordinates improve estimation over ordinary temporal models, PCA, and simple network features.

The working question is:

> Do departures from expected research activity propagate through a scientific relationship graph, and do those dynamics provide advance information about sustained adoption across research communities?

This gives the original Fourier intuition a measurable dynamical assumption. The graph specifies a candidate geometry; future prediction tests whether that geometry describes the evolution of the signal. The proposition remains untested.

## Materials

- [Traffic research plan](traffic-research-plan.md): hypotheses, planned experiments, validation gates, data candidates, references, and an expandable decision log for the traffic-wave study.
- [Literature review](literature-review.md): primary sources, overlap with existing work, data availability, and limits of the search.
- [Study design](study-design.md): hypotheses, mathematical models, outcomes, controls, evaluation, and stopping criteria.
- [Diagnostic script](audit_current_design.py): reproduces two limitations of the existing experiment using the frozen inputs.
- [Diagnostic evidence](evidence/current_design.json) and [cutoff comparisons](evidence/target_comparison.csv): generated results for assessing the current design.

Run the local diagnostics with:

```bash
uv run python research/2026-10/audit_current_design.py
```

## Findings that change the design

The existing midband filter has 99.16% of its squared Frobenius norm in its projection onto a scalar identity matrix. This operator diagnostic measures how closely the filter resembles a uniform rescaling. It helps explain why the filtered score closely resembles raw momentum.

The existing persistent-shift outcome includes an observed pre-cutoff component. With the same 74 eligible topics, a target defined entirely as mean 2023-2025 CLR minus 2022 CLR reduces the spectral composite correlation from 0.517 to 0.236. The growth baseline falls from 0.475 to 0.089. This is an exploratory change of estimand, and it requires separate validation.

The earlier three-year shift remains a valid descriptive measure. Its interpretation combines continuation of an established trend with subsequent movement. A forecasting study should report those components separately.

## Immediate next experiment

First build a small falsification benchmark: independent trends, shared external shocks, and graph-mediated propagation. Compare temporal baselines, graph filters, and equal-capacity models in random and PCA bases. Then use an existing temporal concept-network benchmark to test whether graph features add information over strong local network baselines.

A bounded paper-level corpus is the following step. The proposed first corpus is robotics with adjacent machine-learning and computer-vision papers; quantum information is a possible external replication. Domain choice should follow a coverage audit. Existing quantum-diffusion work makes a domain change alone an insufficient novelty claim.

The public article and its build pipeline have not been changed. These files record a research proposal and exploratory checks; they contain no validated new forecasting result.

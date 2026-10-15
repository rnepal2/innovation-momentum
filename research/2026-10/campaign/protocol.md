# Experiment protocol

Frozen before acquiring paper-level outcomes on 2026-10-14. Changes after observing results must be recorded in the ledger.

## Data

Retrieve English-language OpenAlex works with abstracts, dated 2010-2025, assigned a primary topic of T10462 (Reinforcement Learning in Robotics) or T10653 (Robot Manipulation and Learning). Retrieve the complete filtered corpus by cursor pagination. Retain IDs, dates, titles, abstracts, author IDs, reference IDs, primary topic and DOI. Record query, retrieval time, checksums and exclusions. This is a retrospective reconstruction using current metadata. Acquisition dates and provider changes remain limitations.

Deduplicate records by OpenAlex ID, DOI and normalized exact title where a sufficiently long title also shares a first author. Represent a concept by an observed two-word phrase; learn vocabulary using only 2010-2015 abstracts. Remove generic scholarly phrases and report concept coverage. No pretrained language model is used.

## Synthetic validation

Simulate independent autoregression, a shared latent shock, graph-polynomial propagation and asymmetric propagation. Compare temporal, factor, graph and learned transition models on later observations. Repeat over seeds. Verify ridge predictions are invariant under a full orthonormal change of basis. Assess random graph controls. Simulation results establish behavior under the simulated assumptions.

## H1: residual propagation

Primary horizon: three months. Sensitivity: one month. Predict transformed monthly concept counts and evaluate squared error and count deviance after retransformation. Predict future corpus exposure using past observations when evaluating conditional rates. Reproduce all rates with the denominator explicitly documented.

Train a local temporal baseline with lagged activity, trend, seasonality and past corpus volume. Estimate residuals using causal forecasts. Compare correction models using own residual history, common factors, co-occurrence relationships, and bibliographic relationships. Fit graph-polynomial coefficients using earlier observed residual forecast errors. Compare a learned ridge transition and factor model. Select regularization on historical validation windows. Include graph label permutations and multiple vocabulary sizes where computationally useful.

Development ends in 2018. Evaluate non-overlapping calendar blocks 2019-2020, 2021-2022 and 2023-2025. Fit all preprocessing and models using only data available at each origin. Use time-block loss differences; monthly predictions within a block are dependent.

## H2: sustained recombination

Use a training-only phrase vocabulary of up to 512 phrases with at least 20 supporting papers before 2016. Candidate pairs have zero or one to three past joint-use papers; analyze these risk sets separately. Both concepts must have sufficient historical support at each origin.

Primary event: at least three unique joint-use papers in the next 24 months, occupying at least two six-month bins, with a demonstrable set of at least three papers whose author-ID sets are pairwise disjoint. This author rule is a conservative operational proxy for adoption by separate groups. Also report the repeated-publication outcome without the author rule.

Training origin: end 2015, outcome 2016-2017. Validation origin: end 2017, outcome 2018-2019. Test origins: end 2019, 2021 and 2023, with outcomes 2020-2021, 2022-2023 and 2024-2025. At each test origin, refit using only earlier complete outcome windows. Hyperparameter and model-family selection use an earlier complete validation window.

Compare popularity, historical pair counts, semantic context similarity, common neighbors, Adamic-Adar, preferential attachment and temporal local features against the same controls plus multiscale spectral features. Use regularized logistic models and a bounded boosted-tree comparator. Retain natural class prevalence. Report average precision, precision at fixed review budgets, Brier score and ROC AUC. Report the base rate, sample size and event count for every test set. Avoid outcome-dependent sampling.

## Iteration and interpretation

Allow data-quality corrections, implementation repairs, and training-only model selection. Mark all changes driven by test outcomes as exploratory and report the original test result alongside follow-up results. No claim of causal diffusion follows from predictive association. Three temporal blocks provide limited independent replication.

If spectral gains vanish after common trends, local network features, or graph randomization, stop escalating spectral complexity and characterize the simpler explanation. If a repeatable effect survives, test a second graph construction and target definition before proposing a wider corpus. The objective is a defensible conclusion, including a negative or mixed finding.

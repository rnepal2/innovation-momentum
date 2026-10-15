# Experiment ledger

## E00: source feasibility

The initial OpenAlex filtered query returned 68,697 eligible robotics works. Cursor page two returned HTTP 401, explicitly requiring an API key. The partial page is excluded from analysis. No key is available in the process environment. The legacy arXiv query endpoint returned HTTP 429. The documented OAI-PMH endpoint succeeded.

Before constructing outcomes, the primary acquisition route was changed to a complete OAI harvest of the arXiv `cs:cs:RO` set, restricted locally to first-submission identifier months in 2010-2025. Cross-listed papers retain their category membership. The harvest respects one request per 3.1 seconds.

This source provides structured author names and abstract text, but no references or resolved author IDs. H1 can test a graph constructed from prior concept incidence; citation-based replication needs additional data. H2 uses normalized full-name sets as its adoption proxy and requires sensitivity to identity rules. Historical abstract revisions and ambiguous names limit interpretation. These constraints are part of the conclusions.

H2 excludes phrase pairs sharing a token, which removes nested or overlapping phrases that can generate trivial co-occurrence. Vocabulary and model settings are selected using the declared historical development periods.

## E01: primary source restored; acquisition completed

The user supplied an OpenAlex credential after the initial authorization failure. The fetcher reads it from the process environment. The original primary query completed with 68,697 works. The arXiv cs.RO harvest completed with 47,775 eligible records. Both downloads have local cached pages and SHA-256 manifests. No credential is stored in the experiment files.

The arXiv corpus remains a source-sensitivity analysis. Its scientific population overlaps OpenAlex. Treating the two results as independent replication would overstate the evidence.

Before model evaluation, implementation review identified a development-boundary error: three-month forecasts originating in late 2018 have targets in 2019. Hyperparameter selection now excludes every development forecast whose target is after December 2018. Evaluation starts with January 2019 origins; the first three-month target is April 2019.

## E02: first results and diagnostic follow-ups

The original model outputs are retained under `results/first_pass/`. H1 graph corrections worsened the OpenAlex temporal baseline; arXiv graph and randomized-graph errors were nearly equal. H2 spectral additions gave modest AP gains for weak pairs in OpenAlex. Zero-pair gains failed in the latest block. arXiv's 38-phrase vocabulary left only ten zero-pair candidates and no events in the latest block.

Follow-ups after these results are exploratory. Add a pooled four-coefficient graph correction to test whether node-specific coefficients overfit. Exclude January-1 records while retaining the original vocabulary as a date-heaping sensitivity. This exclusion also removes genuine January-1 papers and changes the population.

Strengthen H2 controls with pair age, degree change, common-neighbor change and weighted-neighbor change. Select each randomized-graph model on the same historical validation window, allowing the same five model configurations as the scientific graph. Store complete baseline and spectral predictions for concept-level resampling. Preserve original results alongside these tests; no newly untouched test is claimed.

The arXiv H1 loop initially ran the same 38-concept three-month specification twice because the 64/128 size caps converged. Deduplicate the specification list. This repeated computation did not create additional observations or change individual losses.

## E03: expanding vocabulary sensitivity

The frozen pre-2016 vocabulary excludes subsequent terminology and progressively exhausts unseen pairs. The arXiv zero-pair risk set shrank to ten pairs in 2023. Add an exploratory rolling-vocabulary run: relearn up to 512 phrases and word-context profiles using records dated by each origin, retaining the same minimum document support, generic filter, pair eligibility, model features and chronological labels. Training examples retain their own historical vocabulary. Feature columns describe pair properties and therefore remain compatible as phrase identities change.

This is a test of fixed-vocabulary saturation and coverage. It cannot establish discovery of a concept before that concept has at least 20 observed papers. All previously inspected periods remain exploratory. No extra external data were acquired for this sensitivity.

## E04: chronological deduplication correction

An independent raw-record check found 715 later-first title/author duplicate encounters in the OpenAlex API ordering. The initial loader retained the first API result. Sorting raw candidates by publication date and ID before deduplication now retains the earliest manifestation under the declared DOI/title rules. This correction affects historical counts, vocabulary support and pair eligibility. Rerun all final experiments from the same downloaded inputs. First-pass files retain their diagnostic status and are excluded from final effect estimates. Interrupted follow-up runs produced no final files.

This rule still cannot link every preprint/journal pair with a changed title or author identity. Pairwise author-disjoint labels reduce some duplicate sensitivity, while incomplete or incorrectly split author identities remain a source of error.

## E05: interpretation checks and stopping boundary

Inspecting phrase lists and ranked pairs identified generic descriptors, URL fragments and inflection variants. Evaluate an explicit lexical-screen sensitivity on the already fitted rankings: remove a declared generic-phrase list, URL tokens and pairs sharing a token after a small published singular mapping. This is a conditional-cohort sensitivity; it leaves training and model coefficients unchanged and cannot validate the remaining pairs as scientific discoveries.

Check probability calibration by fitting a one-variable logistic recalibrator to predictions and completed outcomes from the immediately preceding test window, then apply it to the next window. For example, 2020-2021 outcomes are available at the end-2021 origin used for 2022-2023 predictions. This is an exploratory deployment correction with causal chronological ordering. It changes probabilities and proper scores; a positive calibration slope preserves rankings.

The synthetic benchmark and paper-level chronological controls directly address the two hypotheses. The larger Science4Cast archive is deferred: its edge-only data cannot test disjoint-author adoption, and a full benchmark rerun would answer an additional question. Hypergraphs, pretrained text models and impact prediction remain untested extensions. There is no justification to add them before the current small spectral increments have clearer semantic and external validation.

## E06: final validation

All corrected models and the synthetic benchmark completed. Recalculated numerical summaries and raw-text checks passed. The four generated figures were inspected. The public article, main analysis source and root README remain unchanged. Fetchers now assemble into temporary files; a failed fetch preserves an accepted corpus. OpenAlex also checks the declared count before replacing its output. Future manifests distinguish assembly time and network-page usage from cached data. The existing acquisition manifests retain the actual completed-download timestamps.

## E07: attribution ablation

The final methodological review identified one remaining attribution issue: the spectral augmentation mixes static geometry and two-year changes. Run a bounded exploratory ablation on the cohort with consistent small AP gains, fixed-vocabulary OpenAlex weak pairs. Compare popularity/context alone, static local network controls, local network changes, static spectral additions, spectral-change additions and their full combination. Each feature set chooses among the same five configurations on the original validation window. This analysis can distinguish predictive contributions within this cohort; it creates no new held-out evidence or causal result.

The ablation completed. Spectral changes alone add approximately 0.09, 0.05 and 0.05 AP percentage points over complete local controls; static heat features recover most of the full-model increment. Ordinary local network changes help in one period and hurt in two. The latest static local-network model also beats the full spectral model. This weakens a dynamic-precursor interpretation. Exact reproduction of the original baseline/full-spectral rows passed. Final validation records 59 metric, checksum and table checks plus 96 raw-text cases; the full test suite passes 22 tests.

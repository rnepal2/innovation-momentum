# Experiment Ledger

## A00: scope and source checks

Read the completed campaign and its negative/mixed results. Reviewed primary structural-diversity and homophily-identification literature. Medical corpus topic counts were inspected without paper contents or future-event labels. Chose both medical topics before acquisition. The protocol and operational ontology are frozen in a local Git commit before downloading the new corpus.

The metadata-vintage limitation is material. A bounded first-version audit will measure part of it; current metadata will not be described as historically verified. The operational ontology has agent curation, with human/expert review pending.

## A01: development semantic check

The initial twenty development-paper matches contained 25 method/task pairs. Agent review judged 17 as stated use and eight as comparison-only, background/motivation, or review discussion. Paper-level agreement (at least one qualifying pair) was 16/20; pair-level precision was 17/25. This is insufficient evidence for an adoption endpoint.

Before constructing endpoint labels or evaluating medical records, add explicit role rules: a title mention or an assertion/evaluation sentence is required; comparison-only mentions and review/survey titles are excluded. This conservative rule sacrifices unmeasured recall. Keep the initial cases, assess twenty fresh development cases, and retain a blank independent-human-review sheet. Neither review is an independent expert validation.

The first role-rule validation found 18 supported pairs out of 24, across twenty new papers. It exposed literal implementation errors: the noun "demonstration" triggered a verb-evidence rule; "non-dexterous" and comparative "over" escaped exclusions. Repair these rules before endpoint construction, preserve both samples, and inspect a third disjoint development sample. All such iterations use development text only. Recall remains unmeasured.

## A02: implementation clarifications before outcomes

Apply the fifty-author cap to the primary independence witness and citation-exposure breadth as well as collaboration edges; large consortia otherwise create arbitrary isolated communities. Leave corpus volume and marginal counts visible. Tie-break paper/community assignments by community ID for deterministic runs. Exclude known future-dated cited papers before both flow and exposure construction.

Secondary endpoints use the frozen primary ranking; their probability scores are not separately calibrated forecasts. Add a fixed exact-binomial recent-versus-previous-year burst ranking, with no selected threshold. Report lead time among correct top-budget alerts as descriptive: changing selected cases cannot establish earlier detection for the same events. Prospective outcomes start after the actual UTC freeze date, with any source lag recorded.

## A03: final development text check and freeze

The third, disjoint development sample contains twenty papers and 24 extracted method/task pairs. Agent reading supports 22 pairs; two pairs in a multi-topic thesis have unverified within-document coupling. Freeze the role extractor at parser version 4. The estimate is conditional positive precision from one AI reviewer. Human review, recall, inter-rater agreement and medical-domain semantic precision remain outstanding. Proceed with explicit asserted-co-use and authorship proxies, preserving this limitation in conclusions.

Robotics development has 67 and 91 eligible pairs at its two origins, with six and eleven primary weak-pair events. The training window misses the ten-event power gate. Freeze every configuration at the declared C = 0.2 logistic fallback before reading medical labels. Medical acquisition completed with 89,619 records and USD 0.0449 reported API cost. Full sampled abstracts and downloaded PDFs stay local; review annotations and identifiers are retained in Git.

## A04: held-out results and version-audit repair

Medical evaluation completed with 48, 69 and 85 weak-pair candidates and 12, 8 and 12 primary events. Exposure, breadth and direction increments vary across periods; no consistent top-budget gain is apparent. Preserve all frozen comparisons and stop predictive model expansion.

The first PDF audit wrongly treated line-wrapped PDF text as separate sentences. This produced universal asserted-role disagreement, including single-version papers. Repair text transport by joining wrapped lines and rejoining end-of-line hyphenation before comparing abstract roles. Preserve the initial diagnostic output. This repair affects the version audit only; the predictive extractor and all backtest outputs remain unchanged. An abstract boundary failure remains missing evidence.

## A05: final checks and stopping decision

The corrected PDF audit extracts fourteen of sixteen abstracts: twelve plain-alias agreements and nine assertion-rule agreements. These discrepancies are mixed text/transport/heuristic evidence, with no claim of five confirmed revisions. Matching yields no admissible controls at any medical origin. Report insufficient overlap and retain all failed comparisons.

The primary comparisons fail the practical gate. Add no model or outcome tuning after this result. Preserve the current prospective ledger and input/development manifests on ordinary reruns. A synthetic no-reference test exposed an empty-matrix normalization error; return zero reference similarity for that boundary case. Actual corpora have references, so this repair changes no saved prediction. The full held-out rerun reproduces the saved metrics and leaves the prospective ledger hash unchanged. Additional tests cover identity-weighted AP ties, PDF transport and feature invariance to future records, including a known future-dated reference.

The next study gate is independent semantic and historical-version validation. No public-article revision or future monitoring is performed.

# Cross-community method adoption

Local protocol, frozen before medical-corpus acquisition and new endpoint construction. This is an exploratory follow-up to the robotics campaign, with a new scientific population. It is not an externally registered confirmatory trial. The public article and prior campaign outputs remain unchanged.

## Questions

1. Does prior citation or collaborator exposure predict sustained method/application use by additional authors, after volume, age, growth, semantic overlap and initial use are controlled?
2. Does exposure across multiple past collaboration communities improve on exposure counts alone?
3. Does the direction of inter-community citation paths improve on static, undirected and reversed-path controls?

Exposure and adoption are observational proxies. Publication text, author resolution and homophily can generate associations without interpersonal influence. Structural diversity has established prior art: [Ugander et al. (2012)](https://doi.org/10.1073/pnas.1116502109). The identification problem follows [Aral et al. (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2799846/). No causal or priority claim is planned.

## Population and Meaning

Use the frozen 2010-2025 OpenAlex robotics corpus for development. Acquire the complete English-language, abstract-bearing 2010-2026-10-14 corpus with primary topics T10052 (Medical Image Segmentation Techniques) or T12422 (Radiomics and Machine Learning in Medical Imaging). Count-only source checks found 65,824 and 23,795 records. Exclude medical records matching robotics IDs, DOIs or long normalized titles before analysis. Maximum 120,000 acquired records and USD 0.15 reported API cost; fail without truncating the risk population if either cap is exceeded.

The operational dictionary in `ontology.json` defines methods, synonym rules, robotics tasks and medical tasks. It is researcher/agent-curated and has no independent expert validation. Acronyms alone and broad terms such as "deep learning" do not qualify. Alias matching uses explicit word boundaries after Unicode, hyphen and whitespace normalization. Method families can overlap. The selected cohort of known methods is subject to survivorship and researcher-selection bias; it does not test discovery of unknown concepts.

Review a deterministic sample of twenty positive method/task matches in robotics development abstracts, without future-event labels, for actual use versus background mentions. Freeze corrections before medical evaluation. Report the reviewer's identity as an AI research assistant and keep a human-review sheet. Meaning remains a limitation even after this check.

Current OpenAlex abstracts and identities are retrospective. Extract arXiv links from medical locations and audit a bounded sample against first-version PDFs when available. Preserve version URLs, dates, checksums, failures and text-match differences. Version access failure limits the historical claim; it does not justify treating current metadata as first-public text. The quantitative analysis remains a current-metadata reconstruction unless full first-version coverage is established.

## Unit and Outcomes

Unit: method/application pair at a dated origin. Both terms require at least twenty past corpus papers. Primary risk set: one to ten past joint-use papers. Report zero-use pairs separately with the same model. Use natural prevalence with no outcome-dependent sampling.

Primary event: in the next 24 calendar months, at least three joint-use papers have nonempty author lists, have no listed author among any past joint-use authors, and admit a set of three pairwise author-disjoint papers. Qualifying future papers must occupy at least two six-month bins. Unknown author lists cannot establish independence. Secondary endpoints: at least one new-author paper; and the sustained event after also excluding every past collaborator of seed authors. These rules establish corpus-observed authorship separation, not independent laboratories or scientific quality.

Method and application ages are first observed dates in the selected corpus and may be left-censored. Repeated pair rows across origins are dependent.

## Past-Only Networks

Build collaboration communities from the past corpus using weighted coauthorship and NetworkX Louvain, seed 42 and resolution 1. Papers with over fifty authors are excluded from collaboration-graph construction and retained in a reported sensitivity label; broad consortia otherwise dominate clique expansion. Freeze each origin's communities for its feature calculations. Isolates remain distinct communities. These are algorithmic, corpus-restricted communities.

Citation exposure is the count of application-only papers citing an earlier method-bearing paper in the corpus or an explicitly resolved canonical method reference. Exclude future-dated references when dates are known. Count distinct exposed authors and their frozen communities outside the seed authors. Missing references yield incomplete exposure measurements. Direct collaborator exposure is computed from past authorship only.

Citation flow points from a cited community toward the citing community. Form a row-normalized transition with self-loops removed. Compute one-, two- and three-step propagation between method-source and application-target community distributions. Compare the directed transition with a symmetrized transition and row-normalized reversed flow. These are mechanistic features with learned predictive weights; the operator alone supplies no evolution law.

## Models and Evaluation

Feature ladder: temporal/popularity/semantic controls; static local-network controls; exposure counts; exposure breadth; undirected paths; directed paths. A reversed-path comparator and ten activity-stratified community-label permutations assess specificity. Keep exposure counts fixed in the breadth nulls. Select each feature set on the same development data.

Use logistic ridge C = 0.05, 0.2, 1 and a bounded depth-two boosted tree (60 iterations, minimum leaf 10). Robotics end-2015 origin trains, end-2017 origin validates, with outcomes through 2019. If either split has fewer than ten primary events or two classes, use fixed C = 0.2 logistic models and report the power problem. Apply the frozen configuration to medical origins end-2019, end-2021 and end-2023. At each medical origin, refit on earlier completed medical and robotics windows. Development configurations remain fixed. Medical test rows may join training only after their outcome window is complete.

Report candidates, events, prevalence, AP, top-20 precision (or all available candidates when fewer), Brier and log loss. Resample whole method and application identities for sensitivity intervals; three temporal blocks and a small fixed ontology limit inference. A proposed practical gate is two additional true events per twenty reviewed pairs in at least two held-out periods, with positive AP increments and stable direction/null comparisons. Passing that numerical gate still requires independent scientific review of the suggestions.

Use exact count/age strata and training-feature nearest-neighbor matching as a descriptive check on breadth at similar exposure volume; report unmatched rows and balance. Do not interpret matched differences causally. Construct prospective predictions only with records actually observed at the freeze timestamp, exclude already-seen IDs from future labels, and begin the outcome window after the frozen publication-date cutoff. No automation or later evaluation is scheduled.

## Validation and Changes

Test date boundaries, aliases, seed-author and collaborator exclusions, triple-existence logic, frozen communities, citation direction, future-data invariance and model-selection chronology. Recompute selected labels directly from raw text and author lists. Verify plots visually. Record implementation repairs and all result-driven changes in `ledger.md`. Data acquisition and historical-version audit can proceed concurrently with the development analysis; held-out medical labels remain uninspected until dictionary review and model choices are frozen.

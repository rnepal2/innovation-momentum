# Literature review: spectral dynamics and scientific change

Search date: 2026-10-14. This is a targeted scoping review to choose a research question. It is not an exhaustive systematic review or a claim of priority. Sources were selected for direct methodological overlap, usable data, or a specific evaluation risk.

## Assessment

Graph-based scientific forecasting is an established research area. Prior work includes prediction of new concept links, graph signal processing for paper disruption, and prediction of diffusion across scientific contexts. The existence of a Fourier representation, a citation graph, or an application to a different domain is insufficient evidence of a new contribution.

The most useful opening is to establish when a measured scientific graph provides predictive structure beyond a topic's own past, common trends, and ordinary network statistics. A study could contribute a falsifiable account of propagation, a careful comparison of representations, or a reliable negative result under strict temporal evaluation. These are proposed contributions; the review does not establish that they are novel.

## Closest scientific forecasting work

| Source | Evidence reviewed | Relevant contribution | Implication for this project |
| --- | --- | --- | --- |
| [Krenn and Zeilinger, 2020, Predicting research trends with semantic and neural networks with an application in quantum physics](https://arxiv.org/abs/1906.06843) | Author preprint abstract and publication record | Represents quantum-physics concepts as a semantic network and predicts research trends through new links. | Concept combinations already provide a concrete unit of scientific change. A concept-link application alone would closely follow this work. |
| [Krenn et al., 2023, Forecasting the future of artificial intelligence](https://www.nature.com/articles/s42256-023-00735-0) | Publisher abstract and indexed excerpts; [author repository](https://github.com/artificial-scientist-lab/FutureOfAIviaAI); [data record](https://zenodo.org/records/7882892) | Science4Cast predicts new AI concept links. The repository provides local network baselines and 18 benchmark specifications. | Reproduce existing baselines before adding spectra. Assess rare positive events with precision and calibration as well as ROC AUC. |
| [Yu and Liang, 2024, A framework for predicting scientific disruption based on graph signal processing](https://doi.org/10.1016/j.ipm.2024.103863) | Publisher search excerpts and author institutional listing; full text inaccessible through the reviewed endpoint | Explicitly applies GSP to prediction of scientific disruption using paper content, context, and structure. | Direct overlap with a broad claim that spectral methods can predict scientific innovation. Detailed feature definitions and validation still require full-text review before a formal novelty statement. |
| [Marwitz et al., 2026, Predicting new research directions in materials science using large language models and concept graphs](https://www.nature.com/articles/s42256-026-01206-y) | Full publisher HTML, especially methods and availability sections | Combines extracted concepts, temporal graph features, and text embeddings for link prediction, with human evaluation of suggestions. [Data](https://doi.org/10.6084/m9.figshare.29315819) and [code](https://github.com/aimat-lab/materials_concepts) are available. | Both graph topology and semantic features need strong baselines. Predicting publication occurrence and judging scientific interest are separate tasks. |
| [Maillart et al., 2026, Forecasting Conceptual Diffusion in Science: The Case of Quantum Computing](https://arxiv.org/html/2606.03919v1) | Full preprint HTML, methods, Tables 2-3, and appendices A-B | Distinguishes reinforcement from diffusion using concept-pair citation environments and diversity features. | A close adjacent project. Its primary benchmark uses a stratified 80/20 split across pair-years; its later-year comparison has incomplete downstream windows. Strict forward testing and complete outcomes remain essential. This source is a preprint. |
| [Tomczak et al., first published 2025, journal issue 2026, Forecasting Research Trends Using Knowledge Graphs and Large Language Models](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aisy.202401124) | Full publisher HTML, experimental section and graph-spectrum discussion | Examines temporal scientific knowledge graphs, graph spectra, and changing distances between concepts in nuclear materials. | Spectral descriptions of scientific knowledge graphs also have prior art. Its graph-spectrum diagnostic should be distinguished from forecasting a signal in a graph Fourier basis. |
| [Tshitoyan et al., 2019, Unsupervised word embeddings capture latent knowledge from materials science literature](https://escholarship.org/uc/item/082091b4) | Institutional publication record, indexed abstract, and [author code](https://github.com/materialsintelligence/mat2vec) | Demonstrates scientific structure and useful associations in literature-derived embeddings. | Supports learning representations from literature. Historical tests should control the text dates used to learn a representation. |

The 2026 papers materially affect the choice of topic. Quantum diffusion and materials concept combinations already have recent studies. Their reported metrics use different outcomes and splits, so their numerical performance cannot serve as a direct target for the proposed experiment.

## Mathematical foundations and necessary assumptions

| Source | Evidence reviewed | What it supplies | Question for this project |
| --- | --- | --- | --- |
| [Shuman et al., 2013, The Emerging Field of Signal Processing on Graphs](https://arxiv.org/abs/1211.0053) | Author preprint abstract and bibliographic record | Graph frequency, filtering, and localized multiscale analysis on irregular domains. | Which graph represents a scientifically meaningful relation, and what does variation over that graph mean? |
| [Hammond, Vandergheynst and Gribonval, 2011, Wavelets on Graphs via Spectral Graph Theory](https://arxiv.org/abs/0912.3848) | Author preprint abstract and publication record | Spectral graph wavelets localize signals across node neighborhoods and scales. | Do local persistent changes carry predictive information that global graph modes miss? |
| [Loukas and Perraudin, 2019, Stationary time-vertex signal processing](https://arxiv.org/abs/1611.00255) | Author preprint abstract and publication record | Joint stationarity relates the graph to the covariance of a time-varying signal. | Does the scientific graph approximately diagonalize held-out covariance, after trends are removed? |
| [Isufi et al., 2019, Forecasting Time Series with VARMA Recursions on Graphs](https://arxiv.org/html/1810.08581v2) | Full author manuscript HTML, stationarity definitions, forecasting construction, and empirical discussion | Graph spectral models reduce estimation complexity under approximate joint stationarity; causal temporal recursions provide forecasts. | Does the graph improve temporal estimation relative to matched independent or factor models? A transform by itself does not answer this question. |
| [Brunton et al., 2022, Modern Koopman Theory for Dynamical Systems](https://arxiv.org/abs/2102.12086) | Author preprint abstract and publication record | Spectral analysis of an evolution operator and the role of DMD as a numerical approximation. | Would a learned temporal operator explain the data better than an operator derived from graph geometry? A useful finite observable representation must be demonstrated. |
| [Benson et al., 2018, Simplicial closure and higher-order link prediction](https://arxiv.org/abs/1802.06916) | Author preprint abstract and publication record | Studies interactions of more than two entities and predicts their later joint occurrence. | Does retaining the set of concepts in each paper add information beyond a graph made by projecting papers into pairwise edges? |

Graph Fourier coordinates describe geometry. Koopman or DMD coordinates describe estimated evolution. They can agree when the evolution respects the graph, but that agreement is an empirical assumption. A useful first test asks whether the graph eigenbasis simplifies held-out lagged covariance and forecasting, after controlling for common temporal variation.

## Outcome definition and evaluation

| Source | Evidence reviewed | Relevant lesson | Design consequence |
| --- | --- | --- | --- |
| [Kleinberg, 2002, Bursty and Hierarchical Structure in Streams](https://www.cs.cornell.edu/home/kleinber/bhs.pdf) | Author-hosted paper, abstract and model description | Burst detection provides a principled temporal comparator. | Compare early alerts against ordinary burst and change-point detectors at the same false-alarm budget. |
| [Blei and Lafferty, 2006, Dynamic Topic Models](https://www.cs.cmu.edu/afs/cs/usr/lafferty/www/pub/dtm.pdf) | Author-hosted paper, introductory model description | Models the changing content of topics over time. | A fixed topic taxonomy and changing scientific content are different objects. Topic-word change is an adjacent outcome and a temporal baseline. |
| [Rotolo, Hicks and Martin, 2015, What Is an Emerging Technology?](https://arxiv.org/abs/1503.00673) | Author preprint abstract and publication record | Treats emergence as a multidimensional concept, including growth and coherence. | Define the observable outcome explicitly. Publication growth alone supports a limited statement about attention. |
| [Boettiger and Hastings, 2012, Early Warning Signals and the Prosecutor's Fallacy](https://arxiv.org/abs/1210.1204) | Author preprint abstract | Shows bias introduced by selecting historical systems because a transition occurred. | Include failed candidates and quiet periods. Choose the risk set before outcomes; assess false alarms across that whole set. |

## Data options verified against provider or author sources

| Data source | Available material | Best use | Constraints |
| --- | --- | --- | --- |
| Existing committed OpenAlex snapshot | 77 topic series, annual counts, metadata, June 2026 export | Reproduce the earlier analysis and diagnose filter/target behavior | Aggregate counts have no paper incidence or relationship measurements. There are 33 annual pre-2023 observations, insufficient for a flexible 77-variable temporal model. |
| [Science4Cast author archive](https://zenodo.org/records/7882892) | Author record lists an approximately 1.6 GB archive of 18 link-prediction datasets | Initial temporal edge-prediction benchmark with published comparators | It is an edge benchmark. Paper IDs, original abstracts, and author-group attribution must be checked before treating it as an adoption dataset. No archive was downloaded during this review. |
| [Materials concept-graph archive](https://doi.org/10.6084/m9.figshare.29315819) | Paper states that processed graphs, feature vectors, models, and metrics are provided | External link-prediction replication | Detailed schema and paper-level provenance have not been audited. Author code has a GPL 3.0+ license; data reuse terms require inspection. |
| [arXiv OAI-PMH](https://info.arxiv.org/help/oa/index.html) | Subject-selective metadata harvesting; version history in `arXivRaw` | A bounded monthly paper corpus in robotics or quantum information | OAI exposes the latest article metadata. Its datestamp is record-modification time. Historical text must be tied to a paper version or described as retrospective. |
| [OpenAlex snapshot](https://help.openalex.org/access/snapshot/) | Free bulk works, references, authors, and topics; JSONL and Parquet formats | Citation or bibliographic-coupling edges and paper-level enrichment | Current exports are large and historically revised. `updated_date` partitions are modification partitions. The default API and snapshot corpora can differ; lock corpus membership and release. |
| [SciSciNet-v2](https://northwestern-cssi.github.io/sciscinet/) and [original data paper](https://www.nature.com/articles/s41597-023-02198-9) | Paper-reference, author-affiliation, funding, and patent linkages; public bulk access | Later validation of adoption or translation beyond publication volume | V2 is built on OpenAlex and therefore shares source dependencies. Large text and embedding files make a small extracted subset preferable. BigQuery access is by request. |

No full external corpus has been acquired or validated in this review. Accessible documentation establishes plausible routes to the data, with the listed limitations.

## Search record and remaining gaps

Primary searches covered these question families:

1. Graph signal forecasting: `Forecasting Time Series VARMA Recursions Graphs`, `Stationary time vertex signal processing`.
2. Prior scientific forecasting: `predict scientific research emergence semantic network`, `Forecasting future artificial intelligence Science4Cast`.
3. Direct spectral overlap: `graph Fourier scientific emergence prediction`, `graph signal processing science of science`, `spectral research trends prediction`.
4. Recent adjacent work: `concept diffusion forecasting scientific quantum computing`, `research directions materials science concept graphs 2026`.
5. Alternative mechanisms and evaluation: `simplicial closure higher order link prediction`, `early warning signals prosecutor fallacy`, `bursty hierarchical structure streams`.
6. Feasible data: provider documentation for arXiv harvesting, OpenAlex snapshots, SciSciNet, and author data archives.

Search results led to publisher pages, author manuscripts, official repositories, and provider documentation. Secondary summaries were used to locate original sources, with substantive claims grounded in the original records listed above. Full reading depth differs across sources and is recorded in each table.

Before claiming a new method, obtain the Yu-Liang full text, inspect the nearest papers' supplements and code, and search citation chains for graph-mode forecasting of scientific activity. The absence of an exact title match in this search does not establish novelty.

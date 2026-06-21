# Innovation Momentum

A quick experiment to check if graph-spectral motion can reveal where AI research is moving before the movement becomes obvious. 

[Read the article](reports/index.html)

The current run studies AI/ML research topics from 1990 through June 2026 using public OpenAlex data. The backtest freezes the signal at the end of 2022, ChatGPT moment, then asks whether pre-2023 topic motion anticipates growth from 2023 through the partial 2026 window.

The first result is useful, but deliberately modest. On the 77-topic AI panel, the best primary-scope holdout feature is raw three-year momentum, with Spearman 0.22 against future growth. The graph Fourier emergence score is close behind at 0.20 and remains positive under the wider any-topic count scope. Rolling three-year cutoffs still favor simple publication-growth baselines, so the spectral view is a complementary signal, not a replacement.

The phrase panel gives a second lesson: modern "agentic AI" language appears late, while older components such as planning, retrieval, dialogue, tool use, and multi-agent systems were visible earlier. That is the main design hint for the next version: build the graph from papers, citations, and semantic neighborhoods instead of topic counts alone.

OpenAlex topic counts are public and reproducible, but they are not a complete model of research dynamics. The current graph is built from topic co-movement and metadata similarity; a stronger version should use citation edges, paper embeddings, author/institution flows, and implementation signals.

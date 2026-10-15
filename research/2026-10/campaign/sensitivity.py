"""Post-result lexical screening and past-window probability recalibration."""

import numpy as np
import pandas as pd
from common import DATA, OUT, save_json
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

GENERIC = {
    "real world",
    "real time",
    "large scale",
    "large number",
    "small number",
    "high level",
    "low level",
    "high dimensional",
    "low dimensional",
    "state art",
    "previous work",
    "prior work",
    "future work",
    "previous studies",
    "wide range",
    "good performance",
    "significant improvement",
    "high performance",
    "better performance",
    "extensive experiments",
    "current state",
}
URL = set("http https www github com org arxiv doi et al eg ie".split())
SINGULAR = dict(
    zip(
        "robots networks algorithms models systems agents tasks policies functions vehicles states actions points forces sensors processes mdps pomdps techniques methods environments problems objects".split(),
        "robot network algorithm model system agent task policy function vehicle state action point force sensor process mdp pomdp technique method environment problem object".split(),
    )
)


def screened(a, b):
    if a in GENERIC or b in GENERIC:
        return False
    aa, bb = set(a.split()), set(b.split())
    if (aa | bb) & URL:
        return False
    return not ({SINGULAR.get(w, w) for w in aa} & {SINGULAR.get(w, w) for w in bb})


def main():
    screening, calibration = [], []
    for source in ["openalex", "arxiv"]:
        for suffix in ["", "_rolling"]:
            data = np.load(DATA / f"{source}_h2{suffix}_predictions.npz")
            for risk in ["zero", "weak"]:
                for year in [2019, 2021, 2023]:
                    prefix = f"{year}_{risk}_"
                    vfile = (
                        f"{source}_vocab{year}_vocabulary.csv"
                        if suffix
                        else f"{source}_vocabulary.csv"
                    )
                    vocabulary = pd.read_csv(OUT / vfile).concept.tolist()
                    ii, jj, y = data[prefix + "i"], data[prefix + "j"], data[prefix + "y"]
                    keep = np.array(
                        [screened(vocabulary[i], vocabulary[j]) for i, j in zip(ii, jj)]
                    )
                    for feature in ["base", "spectral"]:
                        p = data[prefix + feature]
                        yy, pp = y[keep], p[keep]
                        screening.append(
                            dict(
                                source=source,
                                vocabulary="rolling" if suffix else "fixed",
                                risk=risk,
                                origin=year,
                                model=feature,
                                original_n=len(y),
                                screened_n=len(yy),
                                events=int(yy.sum()),
                                ap=average_precision_score(yy, pp) if yy.sum() else np.nan,
                                p20=yy[np.argsort(-pp, kind="stable")[:20]].mean()
                                if len(yy)
                                else np.nan,
                            )
                        )
                        if year == 2019:
                            continue
                        old = f"{year - 2}_{risk}_"
                        old_y, old_p = data[old + "y"], data[old + feature]
                        if len(np.unique(old_y)) < 2:
                            continue

                        def transform(q):
                            return logit(np.clip(q, 1e-7, 1 - 1e-7)).reshape(-1, 1)

                        model = LogisticRegression(C=1000, max_iter=500).fit(
                            transform(old_p), old_y
                        )
                        calibrated = model.predict_proba(transform(p))[:, 1]
                        calibration.append(
                            dict(
                                source=source,
                                vocabulary="rolling" if suffix else "fixed",
                                risk=risk,
                                origin=year,
                                model=feature,
                                coefficient=model.coef_[0, 0],
                                intercept=model.intercept_[0],
                                raw_brier=brier_score_loss(y, p),
                                calibrated_brier=brier_score_loss(y, calibrated),
                                raw_logloss=log_loss(y, p, labels=[0, 1]),
                                calibrated_logloss=log_loss(y, calibrated, labels=[0, 1]),
                            )
                        )
    pd.DataFrame(screening).to_csv(OUT / "lexical_screen.csv", index=False)
    pd.DataFrame(calibration).to_csv(OUT / "past_window_calibration.csv", index=False)
    save_json(
        OUT / "lexical_rules.json",
        dict(
            generic_phrases=sorted(GENERIC),
            url_tokens=sorted(URL),
            singular_mapping=SINGULAR,
            interpretation="Exploratory restriction of already fitted rankings; no refitting. This screen does not establish scientific distinctness or synonym resolution.",
        ),
    )


if __name__ == "__main__":
    main()

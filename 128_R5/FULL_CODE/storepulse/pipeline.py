"""Leakage-conscious R5 classification and emerging-issue pipeline."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import f1_score

NON_ISSUE = {"NI01", "NI02", "NI03"}
SEVERITIES = ["low", "medium", "high", "critical"]

CRITICAL_PATTERNS = re.compile(
    r"\b(hospital|hospitalised|hospitalized|allergic reaction|anaphyl|could not breathe|can't breathe|cannot breathe|"
    r"vomit|vomiting|food poisoning|poisoning|injur(?:y|ed)|stitches|stitch|bleeding|assault|hit me|slapped|"
    r"harassed|sexual harassment|undeclared allergen|allergen not declared|not mentioned on label|not on label)\b",
    re.I,
)

# Policy floors from the official taxonomy; applied after model prediction.
HIGH_FLOOR = {"IT04", "IT06", "IT08", "IT10", "IT16", "IT17", "IT18"}
MEDIUM_FLOOR = {"IT27"}


def clean_text(value: Any) -> str:
    """Normalise noisy text while retaining useful multilingual Latin-script cues."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).lower().replace("\u200b", " ")
    text = text.replace("[inaudible]", " inaudible ")
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_text_features() -> FeatureUnion:
    # Word features capture phrases; character features are resilient to typos and transliteration.
    return FeatureUnion([
        ("word", TfidfVectorizer(
            preprocessor=clean_text, analyzer="word", ngram_range=(1, 2),
            min_df=2, max_features=30000, sublinear_tf=True, strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b")),
        ("char", TfidfVectorizer(
            preprocessor=clean_text, analyzer="char_wb", ngram_range=(3, 5),
            min_df=2, max_features=40000, sublinear_tf=True, strip_accents="unicode")),
    ])


def make_issue_model() -> Pipeline:
    return Pipeline([("features", make_text_features()),
                     ("model", LinearSVC(C=1.15, class_weight="balanced"))])


def make_action_model() -> Pipeline:
    return Pipeline([("features", make_text_features()),
                     ("model", LogisticRegression(C=2.0, max_iter=1200, class_weight="balanced", solver="liblinear"))])


def make_severity_model() -> Pipeline:
    return Pipeline([("features", make_text_features()),
                     ("model", LogisticRegression(C=1.4, max_iter=1200, class_weight="balanced", solver="lbfgs"))])


def _taxonomy_map(taxonomy: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(r.issue_type_id): {
        "issue_type": str(r.issue_type), "owner_function": str(r.owner_function),
        "policy_severity_floor": str(r.policy_severity_floor).lower(),
        "is_actionable_category": int(r.is_actionable_category),
        "description": str(r.description),
    } for r in taxonomy.itertuples()}


def train_models(feedback: pd.DataFrame, labels: pd.DataFrame, taxonomy: pd.DataFrame) -> dict[str, Any]:
    """Train only on the supplied public labelled sample; no hidden labels are read."""
    joined = labels.merge(feedback[["feedback_id", "text"]], on="feedback_id", how="inner", validate="one_to_one")
    if len(joined) < 100:
        raise ValueError(f"Expected labelled training records; found {len(joined)} joined rows")
    texts = joined["text"].fillna("").astype(str)
    # Build the expensive sparse text matrix once and share it across the three classifiers.
    features = make_text_features()
    X = features.fit_transform(texts)
    issue_model = LinearSVC(C=1.15, class_weight="balanced").fit(X, joined["issue_type_id"].astype(str))
    action_model = LinearSVC(C=0.8, class_weight="balanced").fit(X, joined["actionable"].astype(int))
    severity_model = LinearSVC(C=0.65, class_weight="balanced").fit(X, joined["severity"].astype(str))
    return {"features": features, "issue": issue_model, "action": action_model, "severity": severity_model,
            "taxonomy": _taxonomy_map(taxonomy), "train_rows": int(len(joined))}


def enforce_policy(issue_ids: np.ndarray, severity: np.ndarray, actionable: np.ndarray,
                   texts: pd.Series, store_ids: pd.Series, taxonomy_map: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply explicit challenge policy floors and conservative actionability checks."""
    issue_ids = np.asarray(issue_ids, dtype=object)
    severity = np.asarray(severity, dtype=object)
    actionable = np.asarray(actionable, dtype=int)
    for i, issue in enumerate(issue_ids):
        meta = taxonomy_map.get(str(issue), {})
        floor = str(meta.get("policy_severity_floor", "low")).lower()
        floor_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(floor, 0)
        current_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(str(severity[i]), 0)
        if floor_rank > current_rank:
            severity[i] = floor
        text = clean_text(texts.iloc[i])
        if issue in HIGH_FLOOR:
            if CRITICAL_PATTERNS.search(text):
                severity[i] = "critical"
            elif severity[i] in ("low", "medium"):
                severity[i] = "high"
        if issue in MEDIUM_FLOOR and severity[i] == "low":
            severity[i] = "medium"
        if issue in NON_ISSUE or int(meta.get("is_actionable_category", 0)) == 0:
            actionable[i] = 0
        else:
            # The label guideline requires a concrete issue and a route. If a store is absent,
            # digital/order issues remain routable; other categories need a store/locality clue.
            digital_routable = issue in {"IT23", "IT24", "IT25", "IT26", "IT27", "IT28", "IT29"}
            known_store = pd.notna(store_ids.iloc[i]) and str(store_ids.iloc[i]).strip() not in ("", "nan", "None")
            specific = len(text.split()) >= 4 and not re.fullmatch(r"(worst|bad|terrible|worst store|very bad)[!. ]*", text)
            actionable[i] = int(bool(actionable[i]) and specific and (known_store or digital_routable))
    return issue_ids, severity, actionable


def predict_records(feedback: pd.DataFrame, models: dict[str, Any], batch_size: int = 2500) -> pd.DataFrame:
    """Predict every row in feedback and enforce policy floors."""
    rows = []
    taxonomy_map = models["taxonomy"]
    for start in range(0, len(feedback), batch_size):
        part = feedback.iloc[start:start + batch_size]
        texts = part["text"].fillna("").astype(str)
        X = models["features"].transform(texts)
        issue = models["issue"].predict(X)
        severity = models["severity"].predict(X)
        action = models["action"].predict(X).astype(int)
        issue, severity, action = enforce_policy(issue, severity, action, texts.reset_index(drop=True),
                                                  part["store_id"].reset_index(drop=True), taxonomy_map)
        rows.append(pd.DataFrame({"feedback_id": part["feedback_id"].astype(str).values,
                                  "issue_type_id": issue, "severity": severity,
                                  "actionable": action.astype(int)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["feedback_id", "issue_type_id", "severity", "actionable"])


def _week_starts(feedback: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(feedback["created_at"], errors="coerce").dt.to_period("W-SUN").dt.start_time.dt.normalize()


def emerging_flags(feedback: pd.DataFrame, predictions: pd.DataFrame, stores: pd.DataFrame,
                   as_of: str | None = None, max_flags: int = 3) -> pd.DataFrame:
    """Causal store-level detector with same-week evidence and volume normalisation.

    A candidate needs >=2 same-week records. It is ranked using the store's trailing
    8-week baseline, store footfall, and chain-wide issue movement to reduce false alarms.
    Only feedback available through the requested week's Sunday is considered.
    """
    columns = ["as_of_week_start", "store_id", "issue_type_id", "confidence", "evidence_feedback_ids"]
    if feedback.empty or predictions.empty:
        return pd.DataFrame(columns=columns)
    fb = feedback.copy()
    fb["created_at_dt"] = pd.to_datetime(fb["created_at"], errors="coerce")
    fb = fb.merge(predictions[["feedback_id", "issue_type_id"]], on="feedback_id", how="left", validate="one_to_one")
    fb["week_start_dt"] = fb["created_at_dt"].dt.to_period("W-SUN").dt.start_time.dt.normalize()
    if as_of:
        target = pd.Timestamp(as_of)
        target = (target - pd.Timedelta(days=target.weekday())).normalize()
        cutoff = target + pd.Timedelta(days=7)
        fb = fb[fb["created_at_dt"] < cutoff]
        target_weeks = [target]
    else:
        target_weeks = sorted(fb["week_start_dt"].dropna().unique())
    fb = fb[fb["store_id"].notna() & (fb["store_id"].astype(str).str.strip() != "")]
    fb = fb[~fb["issue_type_id"].isin(NON_ISSUE)]
    if fb.empty:
        return pd.DataFrame(columns=columns)
    store_meta = stores.copy()
    store_meta["store_id"] = store_meta["store_id"].astype(str)
    footfall = dict(zip(store_meta["store_id"], pd.to_numeric(store_meta["avg_weekly_footfall"], errors="coerce").fillna(1).clip(lower=1)))
    store_names = dict(zip(store_meta["store_id"], store_meta.get("store_name", store_meta["store_id"])))
    fb["store_id"] = fb["store_id"].astype(str)
    counts = fb.groupby(["store_id", "issue_type_id", "week_start_dt"]).size().rename("count").reset_index()
    evidence = fb.groupby(["store_id", "issue_type_id", "week_start_dt"])["feedback_id"].apply(lambda x: ";".join(dict.fromkeys(x.astype(str)))).to_dict()
    chain = fb.groupby(["issue_type_id", "week_start_dt"]).size().to_dict()
    count_map = {(str(r.store_id), str(r.issue_type_id), pd.Timestamp(r.week_start_dt)): int(r.count)
                 for r in counts.itertuples(index=False)}
    all_weeks = sorted(fb["week_start_dt"].dropna().unique())
    week_idx = {pd.Timestamp(w): i for i, w in enumerate(all_weeks)}
    keys = list(counts[["store_id", "issue_type_id"]].drop_duplicates().itertuples(index=False, name=None))
    rows = []
    for week in target_weeks:
        week = pd.Timestamp(week).normalize()
        if week not in week_idx:
            continue
        wi = week_idx[week]
        if wi < 2:
            continue
        candidates = []
        current = counts[counts["week_start_dt"] == week]
        for r in current.itertuples(index=False):
            cur = int(r.count)
            if cur < 2:
                continue
            hist_start = max(0, wi - 8)
            hist_weeks = all_weeks[hist_start:wi]
            hist_vals = [count_map.get((str(r.store_id), str(r.issue_type_id), pd.Timestamp(hw)), 0) for hw in hist_weeks]
            mean = float(np.mean(hist_vals)) if hist_vals else 0.0
            sd = float(np.std(hist_vals)) if hist_vals else 0.0
            threshold = mean + 2.0 * max(sd, np.sqrt(max(mean, 0.0)), 0.75)
            current_rate = cur / float(footfall.get(r.store_id, 1)) * 10000
            hist_rates = [v / float(footfall.get(r.store_id, 1)) * 10000 for v in hist_vals]
            base_rate = float(np.mean(hist_rates)) if hist_rates else 0.0
            chain_hist = [int(chain.get((r.issue_type_id, hw), 0)) for hw in hist_weeks]
            chain_mu = float(np.mean(chain_hist)) if chain_hist else 0.0
            chain_sd = float(np.std(chain_hist)) if chain_hist else 0.0
            chain_cur = int(chain.get((r.issue_type_id, week), 0))
            chain_spike = chain_cur > chain_mu + 2.5 * max(chain_sd, np.sqrt(max(chain_mu, 0.0)), 1.0)
            # Require either a local count spike or a first/re-emerging cluster; discount chain-wide surges.
            local_spike = cur >= threshold or (mean <= 0.25 and cur >= 2)
            if not local_spike:
                continue
            ratio = current_rate / max(base_rate, 0.05)
            growth_score = np.log1p(max(0.0, ratio - 1.0))
            z_score = (cur - mean) / max(sd, np.sqrt(max(mean, 0.0)), 0.75)
            score = max(0.0, z_score) + min(3.0, growth_score) + (0.4 if mean <= 0.25 else 0.0) - (1.2 if chain_spike else 0.0)
            if score <= 0.65:
                continue
            ids = evidence.get((r.store_id, r.issue_type_id, week), "").split(";")
            if len(set(ids)) < 2:
                continue
            candidates.append({"as_of_week_start": week.strftime("%Y-%m-%d"), "store_id": r.store_id,
                               "issue_type_id": r.issue_type_id, "confidence": round(float(1 - np.exp(-max(0.0, score) / 3)), 3),
                               "evidence_feedback_ids": ";".join(ids[:8]), "_score": score,
                               "_name": str(store_names.get(r.store_id, r.store_id))})
        candidates.sort(key=lambda x: (-x["_score"], x["_name"], x["issue_type_id"]))
        rows.extend({k: v for k, v in c.items() if not k.startswith("_")} for c in candidates[:max_flags])
    return pd.DataFrame(rows, columns=columns)


def load_pack(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = Path(data_dir)
    required = ["feedback.csv", "feedback_labels_train.csv", "issue_taxonomy.csv", "store_master.csv", "week_calendar.csv"]
    missing = [x for x in required if not (d / x).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required R5 files in {d}: {', '.join(missing)}")
    fb = pd.read_csv(d / "feedback.csv", dtype={"store_id": str})
    labels = pd.read_csv(d / "feedback_labels_train.csv", dtype={"feedback_id": str})
    taxonomy = pd.read_csv(d / "issue_taxonomy.csv")
    stores = pd.read_csv(d / "store_master.csv", dtype={"store_id": str})
    calendar = pd.read_csv(d / "week_calendar.csv")
    return fb, labels, taxonomy, stores, calendar

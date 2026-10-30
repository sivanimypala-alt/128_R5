"""StorePulse AI demo API. Optional OpenAI responses are server-side only."""
from __future__ import annotations
import os, re, json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from flask import Flask, jsonify, render_template, request
from storepulse.pipeline import load_pack, train_models, predict_records, emerging_flags, NON_ISSUE

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("STOREPULSE_DATA_DIR", ROOT / "data"))
SUBMISSION_DIR = ROOT / "submission"
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
_cache = {"signature": None, "feedback": None, "predictions": None, "stores": None, "taxonomy": None, "flags": None, "models": None}


def _load():
    files = [DATA_DIR / n for n in ("feedback.csv", "feedback_labels_train.csv", "issue_taxonomy.csv", "store_master.csv", "week_calendar.csv")]
    signature = tuple((str(p), p.stat().st_mtime_ns if p.exists() else 0) for p in files)
    if _cache["signature"] == signature and _cache["feedback"] is not None:
        return
    if not all(p.exists() for p in files):
        _cache.update(signature=signature, feedback=pd.DataFrame(), predictions=pd.DataFrame(), stores=pd.DataFrame(), taxonomy=pd.DataFrame(), flags=pd.DataFrame(), models=None)
        return
    feedback, labels, taxonomy, stores, _ = load_pack(DATA_DIR)
    models = train_models(feedback, labels, taxonomy)
    predictions_path = SUBMISSION_DIR / "R5_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        if len(predictions) != len(feedback) or not set(feedback.feedback_id.astype(str)).issubset(set(predictions.feedback_id.astype(str))):
            predictions = predict_records(feedback, models)
    else:
        predictions = predict_records(feedback, models)
    flags_path = SUBMISSION_DIR / "R5_emerging_flags.csv"
    flags = pd.read_csv(flags_path) if flags_path.exists() else emerging_flags(feedback, predictions, stores)
    _cache.update(signature=signature, feedback=feedback, predictions=predictions, stores=stores, taxonomy=taxonomy, flags=flags, models=models)


def _frame():
    _load()
    fb = _cache["feedback"]
    if fb is None or fb.empty:
        return pd.DataFrame()
    pred = _cache["predictions"]
    return fb.merge(pred, on="feedback_id", how="left", suffixes=("", "_pred"))


def _safe_float(value, default=0.0):
    try: return float(value)
    except (TypeError, ValueError): return default

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/overview")
def overview():
    df = _frame()
    if df.empty:
        return jsonify({"ready": False, "message": "Add the official CSV files to the project data/ folder, then refresh.", "total": 0})
    stores = _cache["stores"]
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    current_week = df["created_at"].max().to_period("W-SUN")
    current = df[df["created_at"].dt.to_period("W-SUN") == current_week]
    previous = df[df["created_at"].dt.to_period("W-SUN") == (current_week - 1)]
    issue_rows = df[~df.issue_type_id.isin(NON_ISSUE)]
    top = issue_rows.groupby("issue_type_id").size().sort_values(ascending=False).head(6)
    tax = _cache["taxonomy"].set_index("issue_type_id") if not _cache["taxonomy"].empty else pd.DataFrame()
    top_issues = [{"id": str(k), "name": str(tax.loc[k, "issue_type"]) if k in tax.index else str(k), "count": int(v)} for k,v in top.items()]
    trend = []
    if not df.empty:
        week_series = df.assign(week=df.created_at.dt.to_period("W-SUN").astype(str)).groupby("week").size().tail(8)
        trend = [{"week": str(k)[5:], "count": int(v)} for k,v in week_series.items()]
    return jsonify({"ready": True, "total": int(len(df)), "stores": int(stores.store_id.nunique()),
        "actionable": int((df.actionable == 1).sum()), "high_risk": int(df.severity.isin(["high", "critical"]).sum()),
        "critical": int((df.severity == "critical").sum()), "channels": df.channel.value_counts().to_dict(),
        "regions": int(stores.region.nunique()), "current_week": str(current_week),
        "week_count": int(df.created_at.dt.to_period("W-SUN").nunique()), "top_issues": top_issues,
        "trend": trend, "recent_week_count": int(len(current)), "previous_week_count": int(len(previous)),
        "emerging_count": int(len(_cache["flags"]))})

@app.get("/api/feedback")
def feedback_list():
    df = _frame()
    if df.empty: return jsonify({"items": [], "total": 0})
    q = request.args.get("q", "").strip().lower()[:120]
    channel = request.args.get("channel", "all")
    severity = request.args.get("severity", "all")
    issue = request.args.get("issue", "all")
    region = request.args.get("region", "all")
    limit = min(max(int(request.args.get("limit", 100)), 1), 250)
    df = df.merge(_cache["stores"][["store_id", "store_name", "region", "format"]], on="store_id", how="left")
    if q: df = df[df.text.fillna("").str.lower().str.contains(re.escape(q), regex=True) | df.feedback_id.str.lower().str.contains(re.escape(q), regex=True)]
    if channel != "all": df = df[df.channel == channel]
    if severity != "all": df = df[df.severity == severity]
    if issue != "all": df = df[df.issue_type_id == issue]
    if region != "all": df = df[df.region == region]
    df = df.sort_values("created_at", ascending=False).head(limit)
    items = []
    for r in df.itertuples(index=False):
        items.append({"id": str(r.feedback_id), "date": str(r.created_at), "channel": str(r.channel), "store": str(getattr(r, "store_name", "") or r.store_id or "Store unknown"), "region": str(getattr(r, "region", "") or "Unknown"), "text": str(r.text), "issue": str(r.issue_type_id), "severity": str(r.severity), "actionable": int(r.actionable), "rating": None if pd.isna(r.rating) else int(r.rating)})
    return jsonify({"items": items, "total": int(len(df))})

@app.get("/api/issues")
def issues():
    df = _frame()
    if df.empty: return jsonify({"items": []})
    stores = _cache["stores"]
    df = df.merge(stores[["store_id", "store_name", "region", "avg_weekly_footfall"]], on="store_id", how="left")
    tax = _cache["taxonomy"].set_index("issue_type_id")
    groups = []
    for issue, part in df[~df.issue_type_id.isin(NON_ISSUE)].groupby("issue_type_id"):
        weekly = part.assign(week=pd.to_datetime(part.created_at).dt.to_period("W-SUN").astype(str)).groupby("week").size().sort_index()
        last, prev = (int(weekly.iloc[-1]) if len(weekly) else 0), (int(weekly.iloc[-2]) if len(weekly)>1 else 0)
        affected = part[part.store_id.notna()].store_id.nunique()
        high = int(part.severity.isin(["high", "critical"]).sum())
        actionable = int((part.actionable == 1).sum())
        meta = tax.loc[issue] if issue in tax.index else {}
        groups.append({"id": str(issue), "name": str(meta.get("issue_type", issue)), "owner": str(meta.get("owner_function", "Review")), "count": int(len(part)), "affected_stores": int(affected), "high_risk": high, "actionable": actionable, "trend": "up" if last > prev else ("down" if last < prev else "flat"), "change": last-prev, "severity": "critical" if (part.severity == "critical").any() else ("high" if high else "medium"), "evidence": str(part.iloc[-1].text)[:220]})
    groups.sort(key=lambda x: (x["severity"] != "critical", x["severity"] != "high", -x["high_risk"], -x["count"]))
    return jsonify({"items": groups})

@app.get("/api/emerging")
def emerging():
    _load()
    df = _cache["flags"]
    if df is None or df.empty: return jsonify({"items": []})
    stores = _cache["stores"]
    tax = _cache["taxonomy"].set_index("issue_type_id")
    out = []
    for r in df.to_dict("records"):
        store = stores[stores.store_id.astype(str) == str(r.get("store_id"))]
        out.append({**r, "store_name": str(store.iloc[0].store_name) if len(store) else str(r.get("store_id")), "region": str(store.iloc[0].region) if len(store) else "Unknown", "issue_name": str(tax.loc[r["issue_type_id"], "issue_type"]) if r.get("issue_type_id") in tax.index else str(r.get("issue_type_id")), "evidence_count": len(str(r.get("evidence_feedback_ids", "")).split(";"))})
    return jsonify({"items": out[:80]})

@app.get("/api/brief")
def brief():
    region = request.args.get("region", "North")
    df = _frame()
    if df.empty: return jsonify({"region": region, "items": [], "summary": "Dataset not loaded."})
    stores = _cache["stores"]
    selected = stores[stores.region.astype(str).str.lower() == region.lower()]
    ids = set(selected.store_id.astype(str))
    part = df[df.store_id.astype(str).isin(ids)]
    if part.empty: return jsonify({"region": region, "items": [], "summary": "No feedback records found for this region."})
    tax = _cache["taxonomy"].set_index("issue_type_id")
    part = part[~part.issue_type_id.isin(NON_ISSUE)].copy()
    part["week"] = pd.to_datetime(part.created_at).dt.to_period("W-SUN").astype(str)
    latest, prior = sorted(part.week.dropna().unique())[-1], (sorted(part.week.dropna().unique())[-2] if part.week.nunique()>1 else None)
    current = part[part.week == latest]
    previous = part[part.week == prior] if prior else part.iloc[0:0]
    cur_counts = current.groupby("issue_type_id").size()
    prev_counts = previous.groupby("issue_type_id").size()
    items=[]
    for issue, n in cur_counts.sort_values(ascending=False).head(5).items():
        rows=current[current.issue_type_id==issue]
        prev_n=int(prev_counts.get(issue,0)); stores_affected=rows.store_id.nunique()
        items.append({"id":issue,"name":str(tax.loc[issue,"issue_type"]) if issue in tax.index else issue,"count":int(n),"previous":prev_n,"trend":"↑" if n>prev_n else ("↓" if n<prev_n else "→"),"affected_stores":int(stores_affected),"owner":str(tax.loc[issue,"owner_function"]) if issue in tax.index else "Review","severity":"critical" if (rows.severity=="critical").any() else ("high" if rows.severity.isin(["high","critical"]).any() else "medium"),"evidence":str(rows.iloc[0].text)[:240]})
    return jsonify({"region": region, "week": latest, "items": items, "summary": f"{region} region: {len(current):,} feedback records in the latest week; {int((current.actionable==1).sum())} marked actionable. Treat feedback as signals to verify, not proof by itself."})

@app.post("/api/ask")
def ask():
    body=request.get_json(silent=True) or {}
    question=str(body.get("question", "")).strip()[:500]
    if not question: return jsonify({"answer":"Enter a question about the available store feedback."}),400
    df=_frame()
    if df.empty: return jsonify({"answer":"Dataset is not loaded. Add the official CSV files to data/ first."})
    # Optional API is strictly server-side. Without a key, answer common questions from the actual dataset.
    api_key=os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            import requests
            region_summary=brief().get_json()
            prompt=("You are StorePulse AI, a cautious retail operations analyst. Answer only from the supplied dataset summary. "
                    "Do not invent counts or claim unverified feedback is proven. Cite issue IDs and state when data is insufficient.\n"
                    f"Question: {question}\nDataset summary: {json.dumps(region_summary, ensure_ascii=False)}")
            response=requests.post("https://api.openai.com/v1/responses",headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},json={"model":os.getenv("OPENAI_MODEL","gpt-4o-mini"),"input":prompt,"max_output_tokens":350,"temperature":0.2},timeout=25)
            response.raise_for_status(); data=response.json()
            answer=data.get("output_text")
            if not answer:
                chunks=[]
                for item in data.get("output",[]):
                    for c in item.get("content",[]):
                        if c.get("type")=="output_text": chunks.append(c.get("text",""))
                answer="\n".join(chunks)
            if answer: return jsonify({"answer":answer,"mode":"AI-assisted"})
        except Exception:
            pass
    q=question.lower()
    if any(x in q for x in ["top issue","most common","most frequent","priority"]):
        res=issues().get_json()["items"][:5]
        return jsonify({"answer":"Top issue signals in the loaded data (verify before acting):\n"+"\n".join(f"• {x['name']} ({x['id']}): {x['count']} records; owner {x['owner']}; {x['affected_stores']} stores." for x in res),"mode":"data-backed fallback"})
    if any(x in q for x in ["critical","safety","high risk","allergen","hygiene"]):
        hot=df[df.severity.isin(["high","critical"])].sort_values("created_at",ascending=False).head(8)
        return jsonify({"answer":f"There are {int(df.severity.isin(['high','critical']).sum())} records classified high/critical in the current prediction file. Recent examples:\n"+"\n".join(f"• {r.issue_type_id} / {r.severity} / {r.store_id or 'store unknown'}: {str(r.text)[:150]}" for r in hot.itertuples()),"mode":"data-backed fallback"})
    return jsonify({"answer":"I can answer from the available records about top issues, high-risk feedback, store trends, owners, and emerging flags. Try: ‘What are the top issues?’ or ‘Show high-risk feedback.’ If OPENAI_API_KEY is configured on the server, broader questions use the optional AI endpoint.","mode":"guided fallback"})

@app.get("/api/health")
def health():
    _load(); df=_cache["feedback"]
    return jsonify({"status":"ok","dataset_loaded":bool(df is not None and not df.empty),"records":int(len(df)) if df is not None else 0,"api_key_configured":bool(os.getenv("OPENAI_API_KEY"))})

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","127.0.0.1"),port=int(os.getenv("PORT","5000")),debug=False)
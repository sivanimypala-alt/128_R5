# StorePulse AI — R5 Voice-of-Store Engine

A full-stack prototype and reproducible R5 pipeline for turning noisy multi-channel store feedback into evidence-backed, prioritised operational actions.

## What it does

- Flask dashboard with overview metrics, feedback inbox, issue intelligence, emerging radar, regional weekly brief, and Ask StorePulse.
- Trains three text classifiers (issue type, severity, actionable) using **only** the public `feedback_labels_train.csv` joined to `feedback.csv`. Word TF-IDF plus character n-grams help with typos and transliterated/code-mixed text.
- Applies explicit severity policy floors for safety, hygiene, cold-chain, allergen, contamination and misconduct categories, plus a critical-risk phrase check.
- Produces `submission/R5_predictions.csv`, `submission/R5_emerging_flags.csv`, and `submission/detector_cmd.txt`.
- Provides the causal re-run detector interface required by the challenge.
- Optional Ask StorePulse API calls happen on the backend only; no key is sent to the browser.

## Data setup

The official CSVs are intentionally **not included in the submission ZIP**. Copy the organisers' files into a local `data/` folder beside this README:

- `feedback.csv`
- `feedback_labels_train.csv`
- `issue_taxonomy.csv`
- `store_master.csv`
- `week_calendar.csv`
- `emerging_incidents_train.csv` (needed by the evaluator)

Use the organisers' `evaluate.py` in the project root. Do not upload data files, hidden labels, API keys, or `.env` to a public repository.

## Run in VS Code on Windows

1. Install Python 3.11 and the VS Code Python extension.
2. Open this project folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt terminal and run `.venv\Scripts\activate.bat`.
5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r FULL_CODE/requirements.txt
   ```
6. Copy the official CSVs into `data/` as listed above.
7. Generate the submission files:

   ```powershell
   python FULL_CODE/predict.py
   ```

8. Start the dashboard:

   ```powershell
   python FULL_CODE/app.py
   ```

9. Open `http://127.0.0.1:5000` in your browser. Stop with `Ctrl+C`.

## Optional API key

Ask StorePulse works with data-backed fallback responses without a key. To enable the optional LLM response path, set the environment variables in the terminal before starting Flask:

```powershell
$env:OPENAI_API_KEY="your-key-here"
$env:OPENAI_MODEL="gpt-4o-mini"
python FULL_CODE/app.py
```

Never put the key in `static/app.js`, HTML, screenshots, or the ZIP. API calls may incur provider charges and can fail; the app falls back to data-backed guidance. The current app does not automatically read `.env`; set the environment variables as shown above.

## Generate and evaluate submissions

From the project root, with the official data in `data/`:

```powershell
python FULL_CODE/predict.py
python evaluate.py --submission submission/R5_predictions.csv --truth data/feedback_labels_train.csv
python -m json.tool score.json
```

Use the real evaluator results in the presentation; do not claim hidden-test performance from public labels. The 3,000 public labels are a small, deliberately over-sampled sample, so public evaluation is useful but does not guarantee hidden performance.

## Required causal detector re-run command

`submission/detector_cmd.txt` contains:

```text
python FULL_CODE/detect.py --feedback {feedback} --as-of {asof} --out {out}
```

Example:

```powershell
python FULL_CODE/detect.py --feedback data/feedback.csv --as-of 2026-07-13 --out week.csv
```

The detector reads the required data files from the folder containing `--feedback`, fits on the public labelled sample, and only considers feedback through the Sunday of the requested week. Flags cite same-week feedback IDs and require at least two records. No dates, store IDs, or feedback IDs are hard-coded. Re-run it on any disguised data folder supplied by the organisers.

## Tests

```powershell
python -m pytest FULL_CODE/tests -q
```

## Architecture

- `FULL_CODE/storepulse/pipeline.py`: cleaning, feature extraction, training, policy floors, predictions, causal emerging detector, data loading.
- `FULL_CODE/predict.py`: full submission generation.
- `FULL_CODE/detect.py`: one-week re-run interface for the evaluator/jury.
- `FULL_CODE/app.py`: Flask API and optional server-side AI request.
- `FULL_CODE/templates/` and `FULL_CODE/static/`: frontend.
- `submission/`: required output CSVs and detector command.

## AI usage and originality disclosure

The project uses scikit-learn TF-IDF and linear classifiers trained on the public labelled sample. The optional natural-language answer path calls the OpenAI Responses API only when a server-side key is configured; without it, the app uses a rule-based, data-backed fallback. This implementation was assembled with AI coding assistance; declare the exact pre-built work and AI coding tools your team used, per the participant guidelines. Third-party dependencies are listed in `FULL_CODE/requirements.txt`.

## Limitations

- Classifier quality is limited by 3,000 public labels and messy code-mixed text; inspect evaluator metrics and error cases.
- Severity is difficult to infer from short text. Policy floors are enforced, but critical-risk phrase rules can miss implicit cases or over-trigger.
- The emerging detector is a heuristic baseline, not a statistically validated incident detector. Validate lead time and flag precision with the official scorer; avoid claiming an incident is confirmed.
- Store IDs are frequently missing. The current detector uses records with a known store ID; it does not yet resolve all store names/localities from text.
- Resolution-impact tracking is not implemented as a real workflow in this version; the dashboard focuses on prioritisation and evidence.
- This app is a local prototype, not production-hardened for multi-user deployment.

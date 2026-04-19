# Insult.ai

An AI tool to estimate insult-case penalty value from case features and insult text.

## Deployment Modes

This repository supports two web modes:

- `Flask` mode (Python backend, local/server runtime)
- `Static Pages` mode (pure frontend, works on GitHub Pages / Cloudflare Pages)

---

## Static Pages Mode (Recommended For Free Cloud)

The static version runs directly in browser with no server runtime.

### Model export (required once after training/update)

```bash
python export_web_model.py
```

This generates:

- `docs/assets/model.json`

### Local preview

```bash
python -m http.server 5500 --directory docs
```

Open: `http://127.0.0.1:5500`

### Deploy to GitHub Pages

1. Push repository to GitHub.
2. Go to `Settings` -> `Pages`.
3. Set source to `Deploy from a branch`.
4. Choose branch (for example `Web`) and folder `docs`.
5. Save, then wait for Pages build.

### Deploy to Cloudflare Pages

1. Create a new Pages project and connect this GitHub repo.
2. Build command: leave empty.
3. Build output directory: `docs`.
4. Deploy.

No Python runtime is needed for the static mode.

---

## Flask Mode (Python backend)

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure these trained files exist in the project root:

- `insult_fine_prediction_model.pkl`
- `insult_fine_prediction_vectorizer.pkl`

Optional (for better Chinese tokenization quality):

```bash
python -m spacy download zh_core_web_sm
```

If they do not exist, run training scripts first.

3. Start the web server:

```bash
python Insult.ai.py
```

4. Open browser:

`http://127.0.0.1:5000`

## Notes

- This tool is for research/demo use, not legal advice.
- The prediction output follows the original formatting logic (`元`, `日`, `月`).
- Static Pages mode uses browser-side inference from `docs/assets/model.json`.

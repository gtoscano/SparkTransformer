# BBC Headlines → Multilingual NER (Classroom Pipeline)

This mini-project demonstrates an end‑to‑end, repeatable NLP workflow:

```
🌐 Scrape BBC (EN/ES/FR) → 🧹 Clean → 🧠 Multilingual NER → 📊 Aggregate/Plot → 💾 JSON/CSV
```

## Quickstart

1) **Install** (ideally in a fresh virtualenv):
```bash
pip install -r requirements.txt
```

2) **Run** (online scraping with polite defaults):
```bash
python bbc_ner_pipeline.py --limit 40 --outdir outputs
```

3) **Offline mode** (use saved HTML copies instead of live requests):
```bash
python bbc_ner_pipeline.py --local_html bbc_en.html bbc_es.html bbc_fr.html --limit 40 --outdir outputs
```

## Outputs

- `outputs/bbc_headlines.csv` – raw headlines with language code
- `outputs/bbc_entities.jsonl` – one NER entity per line (JSONL)
- `outputs/bbc_entities.csv` – same as CSV
- `outputs/bbc_entity_counts.csv` – aggregated counts by language & entity type
- `outputs/entity_totals.png` – bar chart of total entities across languages

## Model

Defaults to a lightweight multilingual checkpoint:
- `Davlan/xlm-roberta-base-ner-hrl`

Swap via `--model` if desired, e.g.:
- `Babelscape/wikineural-multilingual-ner`

## Ethics & Etiquette

- Follow each site’s robots.txt and Terms of Use.
- Limit requests (`--limit`) and keep a delay (script includes a small sleep).
- Prefer offline mode in class to avoid hitting sites repeatedly.
- Cite the BBC as your data source when showing outputs.

## Classroom Tips

- Show the pipeline live with `--limit 10` for speed.
- Then switch to offline mode with pre-saved HTML for reproducible results.
- Ask students to extend:
  - Add language detection
  - Compare two NER models
  - Plot per‑language top entity types
  - Export to a dashboard (e.g., Streamlit)

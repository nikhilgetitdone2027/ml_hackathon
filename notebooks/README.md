# Notebooks Directory (`notebooks/`)

This directory is designated for exploratory analysis, interactive experiments, error analysis, and visualization notebooks.

---

## Suggested Notebooks

- **Exploratory Data Analysis (EDA):** Inspect entity record distributions, field nullity, character lengths, and country distributions across `source1.tsv`, `source2.tsv`, and `source3.tsv`.
- **Noise & Abbreviation Analysis:** Analyze common noise patterns, typographical variations, and domain abbreviations in business names and addresses.
- **Error Analysis & Diagnostics:** Inspect false positives, false negatives, and competition margins from `candidate_pairs.tsv` and `matching_results.tsv`.

---

## Running Notebooks

Ensure your virtual environment is active and Jupyter is installed:

```bash
# Activate environment
.venv\Scripts\activate   # Windows

# Install jupyter if needed
pip install jupyter ipykernel

# Launch Jupyter Lab or Notebook
jupyter lab
```

You can also create and run `.ipynb` notebooks directly inside VS Code with the Jupyter extension.

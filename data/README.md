# Data Directory (`data/`)

This directory holds the raw input datasets required by the Amazon ML Hackathon 2026 Business Entity Resolution pipeline.

> **Note:** Raw datasets (`*.tsv`, `*.csv`, `*.json`) are intentionally ignored by `.gitignore` to avoid committing large binary/data files to Git. Place the competition files in this folder on your local machine.

---

## Required Files & Schemas

### 1. `source1.tsv` (Reference Records)
- **Role:** Deduplicated reference entity records. Each record represents a distinct real-world business entity.
- **Format:** Tab-separated values (TSV), UTF-8 encoded.
- **Required Columns:**
  - `entity_id` *(str)*: Unique identifier for the S1 reference record (e.g. `s1_0001`).
  - `business_name` *(str)*: Official business name.
  - `business_address` *(str)*: Street address, locality, postal code.
  - `country` *(str)*: ISO-2 or normalized country code (e.g. `US`, `GB`, `IN`).

### 2. `source2.tsv` (Fragmented/Noisy Records - Vendor A)
- **Role:** Vendor A records containing noisy names, partial addresses, typos, or missing fields that may map to an S1 entity.
- **Format:** Tab-separated values (TSV), UTF-8 encoded.
- **Required Columns:** Same schema as `source1.tsv` (`entity_id`, `business_name`, `business_address`, `country`).

### 3. `source3.tsv` (Fragmented/Noisy Records - Vendor B)
- **Role:** Vendor B records containing noisy names and addresses from an independent vendor source.
- **Format:** Tab-separated values (TSV), UTF-8 encoded.
- **Required Columns:** Same schema as `source1.tsv` (`entity_id`, `business_name`, `business_address`, `country`).

### 4. `ground_truth.tsv` (Training Labels)
- **Role:** Ground-truth training mappings linking S1 entities to matching S2/S3 records.
- **Format:** Tab-separated values (TSV), UTF-8 encoded.
- **Required Columns:**
  - `source1_entity_id` *(str)*: S1 reference entity ID.
  - `matched_ids` *(str)*: Comma-separated list of matching S2/S3 entity IDs (e.g., `s2_1042,s3_8911`), or empty/blank for singletons with no matches.

---

## Sample Data Layout

```
data/
├── .gitkeep                 # Ensures directory is tracked by git
├── README.md                # This specification file
├── source1.tsv              # (User supplied) Reference records
├── source2.tsv              # (User supplied) Noisy records S2
├── source3.tsv              # (User supplied) Noisy records S3
└── ground_truth.tsv         # (User supplied) Training labels
```

---

## Verification & Smoke Test

To verify that your data files are recognized and properly structured before running the full pipeline:

```bash
# Windows PowerShell
.\scripts\run_pipeline.ps1 -SmokeTest

# Or via Python module directly:
python -m src.utils --smoke-test --config configs\config.yaml
```

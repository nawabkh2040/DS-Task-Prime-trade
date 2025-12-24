Overview

This repository contains an analysis pipeline to explore relationships between Hyperliquid trader performance and the Bitcoin Fear & Greed Index.

Files
- `historical_data.csv`: raw historical trades (provided)
- `fear_greed_index.csv`: raw sentiment index (provided)
- `scripts/clean_and_merge.py`: script to clean data and produce daily merged outputs
- `data/clean_historical.csv`, `data/clean_sentiment.csv`, `data/merged_by_date.csv`: outputs (created by running the script)
- `notebooks/`: Jupyter notebook `report.ipynb` will contain full EDA and modeling

Quick start

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the cleaning script:

```bash
python scripts/clean_and_merge.py --historical historical_data.csv --sentiment fear_greed_index.csv --out_dir data
```

3. Open `notebooks/report.ipynb` in VSCode or Jupyter and run cells.


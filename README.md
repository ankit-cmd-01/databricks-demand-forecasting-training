# Demand Forecasting on Databricks

Supply-chain demand forecasting pipeline: raw CSV -> medallion tables -> feature store -> CatBoost model -> predictions.
Everything runs as plain `.py` files, one entrypoint per stage, so each stage can be a Databricks Job task.

## Structure

```
databricks-demand-forecasting-training/
├── config/
│   ├── config.yml          catalog, schemas, table names, paths, split and model settings
│   └── dq_rules.json       data quality rules used by the Silver step
├── src/
│   ├── common/             config_loader.py (reads config.yml), spark_session.py
│   ├── 1_data_engineering/
│   │   ├── bronze.py           raw CSV -> Bronze Delta table
│   │   ├── silver.py           DQ rules; clean rows -> Silver, failed rows -> config.dq_failed_records
│   │   ├── gold.py             one row per product + city + date
│   │   ├── gold_feature.py     calendar, lag and rolling features
│   │   ├── feature_store.py    registers or refreshes the Feature Store table
│   │   └── de_entrypoint.py    runs all of the above in order
│   ├── 2_model_training/
│   │   ├── data_splitting.py          chronological split: train / validation / inference holdout
│   │   ├── model_training.py          CatBoost + MLflow + Unity Catalog registration
│   │   └── model_train_entrypoint.py
│   ├── 3_inference/
│   │   ├── model_predictions.py       latest model version scores the validation set
│   │   └── inference_entrypoint.py
│   └── requirements.txt
└── README.md
```

## How the pieces fit

- Only the entrypoint files read `config.yml`. They build the table names and pass plain values into the classes.
- Class files do the actual work and never see the config, so they work for any catalog or schema.
- Each entrypoint adds `src/` to `sys.path`, which is how `common` is found.

## Data flow

1. **Stage 1** `de_entrypoint.py`: CSV -> Bronze -> Silver -> Gold -> features -> Feature Store table.
2. **Stage 2** `model_train_entrypoint.py`: the Feature Store table is split by date into train, validation and an inference holdout. CatBoost is trained, logged to MLflow and registered as a new Unity Catalog model version.
3. **Stage 3** `inference_entrypoint.py`: the latest registered version scores the validation set and writes the predictions table.

## Running

Each entrypoint is a Job task of type Python script, run in this order:
`de_entrypoint.py` -> `model_train_entrypoint.py` -> `inference_entrypoint.py`.
Install `src/requirements.txt` on the job compute or environment.

## Tables

All tables live in the catalog set in `config.yml` (`catalog_mle` by default), in the `bronze`, `silver`, `gold`, `config` and `ml` schemas.
Model: `<catalog>.ml.demand_forecasting_catboost`.
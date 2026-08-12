-- MLflow gets its own database in the same Postgres instance (its backend
-- store for experiment/run/registry metadata is unrelated to the
-- application's `houses`/`predictions` tables and should not live in the
-- same schema).
CREATE DATABASE mlflow;

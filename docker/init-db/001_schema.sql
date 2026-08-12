-- "Data warehouse" schema for the house-price MLOps project.
-- houses: the feature+label table a data engineer would have ETL'd from
-- upstream source systems (listings, county records, etc).
CREATE TABLE IF NOT EXISTS houses (
    id              BIGSERIAL PRIMARY KEY,
    med_inc         DOUBLE PRECISION NOT NULL,
    house_age       DOUBLE PRECISION NOT NULL,
    ave_rooms       DOUBLE PRECISION NOT NULL,
    ave_bedrms      DOUBLE PRECISION NOT NULL,
    population      DOUBLE PRECISION NOT NULL,
    ave_occup       DOUBLE PRECISION NOT NULL,
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    med_house_val   DOUBLE PRECISION NOT NULL,   -- label / target
    source          VARCHAR(32) NOT NULL DEFAULT 'historical_batch', -- historical_batch | live_stream
    ingested_at     TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_houses_ingested_at ON houses (ingested_at);
CREATE INDEX IF NOT EXISTS idx_houses_source ON houses (source);

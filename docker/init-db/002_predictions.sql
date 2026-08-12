-- Logs every inference request the serving app handles.
-- actual_value stays NULL until a real sale closes and we learn the true
-- price - that delayed-label pattern is normal for real estate (and for
-- most real-world regression problems: you predict now, find out later).
-- This table is what model-performance monitoring and drift checks read from.
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    requested_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    med_inc         DOUBLE PRECISION NOT NULL,
    house_age       DOUBLE PRECISION NOT NULL,
    ave_rooms       DOUBLE PRECISION NOT NULL,
    ave_bedrms      DOUBLE PRECISION NOT NULL,
    population      DOUBLE PRECISION NOT NULL,
    ave_occup       DOUBLE PRECISION NOT NULL,
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    bedrms_ratio    DOUBLE PRECISION NOT NULL,
    predicted_value DOUBLE PRECISION NOT NULL,
    model_name      VARCHAR(128) NOT NULL,
    model_version   VARCHAR(16) NOT NULL,
    actual_value    DOUBLE PRECISION  -- filled in later, once known
);

CREATE INDEX IF NOT EXISTS idx_predictions_requested_at ON predictions (requested_at);

-- Migration 013: Lineage-safe timestamps for predictions and player availability
-- Purpose:
--   1) Preserve immutable first-seen timestamps
--   2) Preserve mutable refresh timestamps separately
--   3) Add semantic event/as-of timestamp columns for PIT audits

-- -----------------------------
-- predictions lineage columns
-- -----------------------------
ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS first_created_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_refreshed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS feature_asof_utc TIMESTAMPTZ;

ALTER TABLE predictions
    ALTER COLUMN first_created_at SET DEFAULT NOW(),
    ALTER COLUMN last_refreshed_at SET DEFAULT NOW();

UPDATE predictions
SET
    first_created_at = COALESCE(first_created_at, created_at),
    last_refreshed_at = COALESCE(last_refreshed_at, created_at)
WHERE first_created_at IS NULL
   OR last_refreshed_at IS NULL;

ALTER TABLE predictions
    ALTER COLUMN first_created_at SET NOT NULL,
    ALTER COLUMN last_refreshed_at SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_predictions_feature_asof_utc
    ON predictions(feature_asof_utc);

CREATE INDEX IF NOT EXISTS idx_predictions_first_created_at
    ON predictions(first_created_at);

CREATE OR REPLACE FUNCTION set_predictions_lineage_columns()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        NEW.first_created_at := COALESCE(NEW.first_created_at, NEW.created_at, NOW());
        NEW.last_refreshed_at := COALESCE(NEW.last_refreshed_at, NOW());
        NEW.created_at := COALESCE(NEW.created_at, NEW.first_created_at, NOW());
        RETURN NEW;
    END IF;

    NEW.first_created_at := COALESCE(OLD.first_created_at, NEW.first_created_at, OLD.created_at, NEW.created_at, NOW());
    NEW.last_refreshed_at := NOW();
    NEW.created_at := COALESCE(NEW.created_at, OLD.created_at, NEW.first_created_at, NOW());
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_set_predictions_lineage_columns ON predictions;
CREATE TRIGGER trg_set_predictions_lineage_columns
BEFORE INSERT OR UPDATE ON predictions
FOR EACH ROW
EXECUTE FUNCTION set_predictions_lineage_columns();

-- ------------------------------------
-- player_availability lineage columns
-- ------------------------------------
ALTER TABLE player_availability
    ADD COLUMN IF NOT EXISTS first_recorded_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_refreshed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS event_recorded_at TIMESTAMPTZ;

ALTER TABLE player_availability
    ALTER COLUMN first_recorded_at SET DEFAULT NOW(),
    ALTER COLUMN last_refreshed_at SET DEFAULT NOW();

UPDATE player_availability
SET
    first_recorded_at = COALESCE(first_recorded_at, recorded_at),
    last_refreshed_at = COALESCE(last_refreshed_at, recorded_at)
WHERE first_recorded_at IS NULL
   OR last_refreshed_at IS NULL;

UPDATE player_availability
SET event_recorded_at = COALESCE(event_recorded_at, first_recorded_at, recorded_at)
WHERE event_recorded_at IS NULL;

ALTER TABLE player_availability
    ALTER COLUMN first_recorded_at SET NOT NULL,
    ALTER COLUMN last_refreshed_at SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_player_availability_event_recorded_at
    ON player_availability(event_recorded_at);

CREATE INDEX IF NOT EXISTS idx_player_availability_first_recorded_at
    ON player_availability(first_recorded_at);

CREATE OR REPLACE FUNCTION set_player_availability_lineage_columns()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        NEW.first_recorded_at := COALESCE(NEW.first_recorded_at, NEW.recorded_at, NOW());
        NEW.last_refreshed_at := COALESCE(NEW.last_refreshed_at, NOW());
        NEW.recorded_at := COALESCE(NEW.recorded_at, NEW.first_recorded_at, NOW());
        NEW.event_recorded_at := COALESCE(NEW.event_recorded_at, NEW.recorded_at, NEW.first_recorded_at, NOW());
        RETURN NEW;
    END IF;

    NEW.first_recorded_at := COALESCE(OLD.first_recorded_at, NEW.first_recorded_at, OLD.recorded_at, NEW.recorded_at, NOW());
    NEW.last_refreshed_at := NOW();
    NEW.recorded_at := COALESCE(NEW.recorded_at, OLD.recorded_at, NEW.first_recorded_at, NOW());
    NEW.event_recorded_at := COALESCE(NEW.event_recorded_at, OLD.event_recorded_at, NEW.recorded_at, OLD.recorded_at, NOW());
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_set_player_availability_lineage_columns ON player_availability;
CREATE TRIGGER trg_set_player_availability_lineage_columns
BEFORE INSERT OR UPDATE ON player_availability
FOR EACH ROW
EXECUTE FUNCTION set_player_availability_lineage_columns();

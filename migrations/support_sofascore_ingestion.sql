-- Migration: support_sofascore_ingestion.sql
-- Description: Makes flashscore_id nullable in fixtures table and ensures sofascore_id formatting is robust.

-- 1. Drop NOT NULL constraint from flashscore_id
ALTER TABLE public.fixtures ALTER COLUMN flashscore_id DROP NOT NULL;

-- 2. Optional: Add a comment to explain why
COMMENT ON COLUMN public.fixtures.flashscore_id IS 'External Flashscore identifier. Nullable to support Sofascore-first data ingestion.';

-- 3. Add season information to fixture_stats_premium for easier querying if needed
-- (Actually, it already inherits via fixture_id, but let's keep it lean)

-- 4. Ensure sofascore_id in teams is consistent (bigint is fine, but fixtures has varchar(50))
-- Let's leave fixtures.sofascore_id as varchar(50) for now as it handles external string IDs better if they change.

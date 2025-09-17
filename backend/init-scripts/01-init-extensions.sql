-- Initialize PostgreSQL extensions and setup
-- This script runs when the PostgreSQL container starts for the first time

-- Create UUID extension for generating UUIDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create additional useful extensions
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text similarity searches
CREATE EXTENSION IF NOT EXISTS "unaccent"; -- For removing accents from text

-- Set timezone
SET timezone = 'UTC';

-- Create a function to update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';
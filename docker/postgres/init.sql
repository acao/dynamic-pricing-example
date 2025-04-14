-- Create database if it doesn't exist
CREATE DATABASE insurance_pricing;

-- Connect to the database
\c insurance_pricing;

-- Create schema for ML models
CREATE SCHEMA IF NOT EXISTS ml_models;

-- Create model versions table
CREATE TABLE IF NOT EXISTS ml_models.model_versions (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_path VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, model_version)
);

-- Create pricing history table
CREATE TABLE IF NOT EXISTS ml_models.pricing_history (
    pricing_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id UUID NOT NULL,
    model_id UUID NOT NULL REFERENCES ml_models.model_versions(model_id),
    input_features JSONB NOT NULL,
    base_premium NUMERIC(10, 2) NOT NULL,
    adjustments JSONB,
    final_premium NUMERIC(10, 2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on pricing history
CREATE INDEX IF NOT EXISTS idx_pricing_history_policy_id ON ml_models.pricing_history(policy_id);
CREATE INDEX IF NOT EXISTS idx_pricing_history_model_id ON ml_models.pricing_history(model_id);

-- Create feature importance table
CREATE TABLE IF NOT EXISTS ml_models.feature_importance (
    feature_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES ml_models.model_versions(model_id),
    feature_name VARCHAR(100) NOT NULL,
    importance NUMERIC(10, 6) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, feature_name)
);

-- Create model metrics table
CREATE TABLE IF NOT EXISTS ml_models.model_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES ml_models.model_versions(model_id),
    metric_name VARCHAR(50) NOT NULL,
    metric_value NUMERIC(10, 6) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, metric_name)
);

-- Create view for active model
CREATE OR REPLACE VIEW ml_models.active_model AS
SELECT * FROM ml_models.model_versions WHERE is_active = TRUE;

-- Create view for pricing statistics
CREATE OR REPLACE VIEW ml_models.pricing_statistics AS
SELECT
    model_id,
    COUNT(*) AS request_count,
    AVG(final_premium) AS avg_premium,
    MIN(final_premium) AS min_premium,
    MAX(final_premium) AS max_premium,
    STDDEV(final_premium) AS stddev_premium,
    DATE_TRUNC('day', created_at) AS request_date
FROM ml_models.pricing_history
GROUP BY model_id, DATE_TRUNC('day', created_at)
ORDER BY DATE_TRUNC('day', created_at) DESC;

-- Create function to register a new model
CREATE OR REPLACE FUNCTION ml_models.register_model(
    p_model_name VARCHAR,
    p_model_version VARCHAR,
    p_model_path VARCHAR,
    p_is_active BOOLEAN,
    p_metrics JSONB
) RETURNS UUID AS $$
DECLARE
    v_model_id UUID;
BEGIN
    -- Insert model record
    INSERT INTO ml_models.model_versions (
        model_name, model_version, model_path, is_active, metrics
    ) VALUES (
        p_model_name, p_model_version, p_model_path, p_is_active, p_metrics
    ) RETURNING model_id INTO v_model_id;
    
    -- If this model is active, deactivate other models
    IF p_is_active THEN
        UPDATE ml_models.model_versions
        SET is_active = FALSE
        WHERE model_id != v_model_id;
    END IF;
    
    -- Insert metrics
    IF p_metrics IS NOT NULL THEN
        FOR metric_name, metric_value IN SELECT * FROM jsonb_each_text(p_metrics) LOOP
            INSERT INTO ml_models.model_metrics (
                model_id, metric_name, metric_value
            ) VALUES (
                v_model_id, metric_name, metric_value::NUMERIC
            );
        END LOOP;
    END IF;
    
    RETURN v_model_id;
END;
$$ LANGUAGE plpgsql;

-- Create function to log a pricing request
CREATE OR REPLACE FUNCTION ml_models.log_pricing_request(
    p_policy_id UUID,
    p_model_id UUID,
    p_input_features JSONB,
    p_base_premium NUMERIC,
    p_adjustments JSONB,
    p_final_premium NUMERIC
) RETURNS UUID AS $$
DECLARE
    v_pricing_id UUID;
BEGIN
    -- Insert pricing record
    INSERT INTO ml_models.pricing_history (
        policy_id, model_id, input_features, base_premium, adjustments, final_premium
    ) VALUES (
        p_policy_id, p_model_id, p_input_features, p_base_premium, p_adjustments, p_final_premium
    ) RETURNING pricing_id INTO v_pricing_id;
    
    RETURN v_pricing_id;
END;
$$ LANGUAGE plpgsql;

-- Create role for the application
CREATE ROLE insurance_pricing_app WITH LOGIN PASSWORD 'password';

-- Grant privileges to the role
GRANT USAGE ON SCHEMA ml_models TO insurance_pricing_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA ml_models TO insurance_pricing_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA ml_models TO insurance_pricing_app;

-- Insert sample data for testing
-- Insert a sample model version
INSERT INTO ml_models.model_versions (model_name, model_version, model_path, is_active, metrics)
VALUES (
    'decision_tree',
    '20250413000000',
    '/app/models/decision_tree_20250413000000.joblib',
    TRUE,
    '{"rmse": 120.5, "mae": 95.2, "r2": 0.85}'
);

-- Get the model ID
DO $$
DECLARE
    v_model_id UUID;
BEGIN
    SELECT model_id INTO v_model_id FROM ml_models.model_versions WHERE model_version = '20250413000000';
    
    -- Insert sample feature importances
    INSERT INTO ml_models.feature_importance (model_id, feature_name, importance)
    VALUES
        (v_model_id, 'driver_age_mean', 0.25),
        (v_model_id, 'vehicle_value_mean', 0.20),
        (v_model_id, 'accident_count', 0.15),
        (v_model_id, 'driver_experience_mean', 0.10),
        (v_model_id, 'vehicle_age_mean', 0.08),
        (v_model_id, 'location_crime_rate_mean', 0.07),
        (v_model_id, 'policy_duration_days', 0.05),
        (v_model_id, 'vehicle_per_driver_ratio', 0.04),
        (v_model_id, 'years_since_last_incident', 0.03),
        (v_model_id, 'incident_severity_score', 0.03);
    
    -- Insert sample metrics
    INSERT INTO ml_models.model_metrics (model_id, metric_name, metric_value)
    VALUES
        (v_model_id, 'rmse', 120.5),
        (v_model_id, 'mae', 95.2),
        (v_model_id, 'r2', 0.85);
    
    -- Insert sample pricing history
    INSERT INTO ml_models.pricing_history (policy_id, model_id, input_features, base_premium, adjustments, final_premium)
    VALUES
        (
            gen_random_uuid(),
            v_model_id,
            '{"driver_age_mean": 35, "vehicle_value_mean": 25000, "accident_count": 0}',
            1000.0,
            '{"credit_factor": 0.9, "territory_factor": 1.1}',
            990.0
        ),
        (
            gen_random_uuid(),
            v_model_id,
            '{"driver_age_mean": 22, "vehicle_value_mean": 15000, "accident_count": 1}',
            1200.0,
            '{"credit_factor": 1.2, "territory_factor": 1.0}',
            1440.0
        ),
        (
            gen_random_uuid(),
            v_model_id,
            '{"driver_age_mean": 45, "vehicle_value_mean": 35000, "accident_count": 0}',
            1100.0,
            '{"credit_factor": 0.8, "territory_factor": 0.9}',
            792.0
        );
END $$;

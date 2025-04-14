-- Create database
CREATE DATABASE IF NOT EXISTS INSURANCE_PRICING;
USE DATABASE INSURANCE_PRICING;

-- Create warehouse
CREATE WAREHOUSE IF NOT EXISTS INSURANCE_PRICING_WH
  WITH WAREHOUSE_SIZE = 'X-SMALL'
  AUTO_SUSPEND = 300
  AUTO_RESUME = TRUE;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS ML_MODELS;
CREATE SCHEMA IF NOT EXISTS PRICING;
CREATE SCHEMA IF NOT EXISTS STAGING;

-- Use warehouse
USE WAREHOUSE INSURANCE_PRICING_WH;

-- Create model versions table
CREATE OR REPLACE TABLE ML_MODELS.MODEL_VERSIONS (
    MODEL_ID VARCHAR(36) DEFAULT UUID_STRING(),
    MODEL_NAME VARCHAR(100) NOT NULL,
    MODEL_VERSION VARCHAR(20) NOT NULL,
    MODEL_PATH VARCHAR(255) NOT NULL,
    IS_ACTIVE BOOLEAN DEFAULT FALSE,
    METRICS VARIANT,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (MODEL_ID),
    UNIQUE (MODEL_NAME, MODEL_VERSION)
);

-- Create feature importance table
CREATE OR REPLACE TABLE ML_MODELS.FEATURE_IMPORTANCE (
    FEATURE_ID VARCHAR(36) DEFAULT UUID_STRING(),
    MODEL_ID VARCHAR(36) NOT NULL,
    FEATURE_NAME VARCHAR(100) NOT NULL,
    IMPORTANCE FLOAT NOT NULL,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (FEATURE_ID),
    FOREIGN KEY (MODEL_ID) REFERENCES ML_MODELS.MODEL_VERSIONS(MODEL_ID),
    UNIQUE (MODEL_ID, FEATURE_NAME)
);

-- Create model metrics table
CREATE OR REPLACE TABLE ML_MODELS.MODEL_METRICS (
    METRIC_ID VARCHAR(36) DEFAULT UUID_STRING(),
    MODEL_ID VARCHAR(36) NOT NULL,
    METRIC_NAME VARCHAR(50) NOT NULL,
    METRIC_VALUE FLOAT NOT NULL,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (METRIC_ID),
    FOREIGN KEY (MODEL_ID) REFERENCES ML_MODELS.MODEL_VERSIONS(MODEL_ID),
    UNIQUE (MODEL_ID, METRIC_NAME)
);

-- Create pricing history table
CREATE OR REPLACE TABLE PRICING.PRICING_HISTORY (
    PRICING_ID VARCHAR(36) DEFAULT UUID_STRING(),
    POLICY_ID VARCHAR(36) NOT NULL,
    MODEL_ID VARCHAR(36) NOT NULL,
    INPUT_FEATURES VARIANT NOT NULL,
    BASE_PREMIUM FLOAT NOT NULL,
    ADJUSTMENTS VARIANT,
    FINAL_PREMIUM FLOAT NOT NULL,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (PRICING_ID),
    FOREIGN KEY (MODEL_ID) REFERENCES ML_MODELS.MODEL_VERSIONS(MODEL_ID)
);

-- Create index on pricing history
CREATE OR REPLACE INDEX IDX_PRICING_HISTORY_POLICY_ID ON PRICING.PRICING_HISTORY(POLICY_ID);
CREATE OR REPLACE INDEX IDX_PRICING_HISTORY_MODEL_ID ON PRICING.PRICING_HISTORY(MODEL_ID);

-- Create staging table for model training data
CREATE OR REPLACE TABLE STAGING.TRAINING_DATA (
    POLICY_ID VARCHAR(36) DEFAULT UUID_STRING(),
    POLICY_DATA VARIANT NOT NULL,
    PREMIUM FLOAT NOT NULL,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (POLICY_ID)
);

-- Create view for active model
CREATE OR REPLACE VIEW ML_MODELS.ACTIVE_MODEL AS
SELECT * FROM ML_MODELS.MODEL_VERSIONS WHERE IS_ACTIVE = TRUE;

-- Create view for pricing statistics
CREATE OR REPLACE VIEW PRICING.PRICING_STATISTICS AS
SELECT
    MODEL_ID,
    COUNT(*) AS REQUEST_COUNT,
    AVG(FINAL_PREMIUM) AS AVG_PREMIUM,
    MIN(FINAL_PREMIUM) AS MIN_PREMIUM,
    MAX(FINAL_PREMIUM) AS MAX_PREMIUM,
    STDDEV(FINAL_PREMIUM) AS STDDEV_PREMIUM,
    DATE_TRUNC('DAY', CREATED_AT) AS REQUEST_DATE
FROM PRICING.PRICING_HISTORY
GROUP BY MODEL_ID, DATE_TRUNC('DAY', CREATED_AT)
ORDER BY DATE_TRUNC('DAY', CREATED_AT) DESC;

-- Create stored procedure to register a new model
CREATE OR REPLACE PROCEDURE ML_MODELS.REGISTER_MODEL(
    MODEL_NAME VARCHAR,
    MODEL_VERSION VARCHAR,
    MODEL_PATH VARCHAR,
    IS_ACTIVE BOOLEAN,
    METRICS VARIANT
)
RETURNS VARCHAR
LANGUAGE JAVASCRIPT
AS
$$
    var model_id = UUID_STRING();
    
    // Insert model record
    var sql_insert = `
        INSERT INTO ML_MODELS.MODEL_VERSIONS (
            MODEL_ID, MODEL_NAME, MODEL_VERSION, MODEL_PATH, IS_ACTIVE, METRICS
        ) VALUES (
            '${model_id}', '${MODEL_NAME}', '${MODEL_VERSION}', '${MODEL_PATH}', ${IS_ACTIVE}, PARSE_JSON('${METRICS}')
        )
    `;
    
    try {
        snowflake.execute({sqlText: sql_insert});
        
        // If this model is active, deactivate other models
        if (IS_ACTIVE) {
            var sql_update = `
                UPDATE ML_MODELS.MODEL_VERSIONS
                SET IS_ACTIVE = FALSE
                WHERE MODEL_ID != '${model_id}'
            `;
            snowflake.execute({sqlText: sql_update});
        }
        
        // Insert metrics
        if (METRICS) {
            var metrics = JSON.parse(METRICS);
            for (var metric_name in metrics) {
                if (metrics.hasOwnProperty(metric_name)) {
                    var metric_value = metrics[metric_name];
                    var sql_metric = `
                        INSERT INTO ML_MODELS.MODEL_METRICS (
                            MODEL_ID, METRIC_NAME, METRIC_VALUE
                        ) VALUES (
                            '${model_id}', '${metric_name}', ${metric_value}
                        )
                    `;
                    snowflake.execute({sqlText: sql_metric});
                }
            }
        }
        
        return model_id;
    } catch (err) {
        return "Error: " + err;
    }
$$;

-- Create stored procedure to log a pricing request
CREATE OR REPLACE PROCEDURE PRICING.LOG_PRICING_REQUEST(
    POLICY_ID VARCHAR,
    MODEL_ID VARCHAR,
    INPUT_FEATURES VARIANT,
    BASE_PREMIUM FLOAT,
    ADJUSTMENTS VARIANT,
    FINAL_PREMIUM FLOAT
)
RETURNS VARCHAR
LANGUAGE JAVASCRIPT
AS
$$
    var pricing_id = UUID_STRING();
    
    // Insert pricing record
    var sql_insert = `
        INSERT INTO PRICING.PRICING_HISTORY (
            PRICING_ID, POLICY_ID, MODEL_ID, INPUT_FEATURES, BASE_PREMIUM, ADJUSTMENTS, FINAL_PREMIUM
        ) VALUES (
            '${pricing_id}', '${POLICY_ID}', '${MODEL_ID}', PARSE_JSON('${INPUT_FEATURES}'), ${BASE_PREMIUM}, PARSE_JSON('${ADJUSTMENTS}'), ${FINAL_PREMIUM}
        )
    `;
    
    try {
        snowflake.execute({sqlText: sql_insert});
        return pricing_id;
    } catch (err) {
        return "Error: " + err;
    }
$$;

-- Create stored procedure to get the active model
CREATE OR REPLACE PROCEDURE ML_MODELS.GET_ACTIVE_MODEL()
RETURNS VARIANT
LANGUAGE JAVASCRIPT
AS
$$
    var sql = `SELECT * FROM ML_MODELS.ACTIVE_MODEL`;
    var stmt = snowflake.createStatement({sqlText: sql});
    var rs = stmt.execute();
    
    if (rs.next()) {
        var model = {
            model_id: rs.getColumnValue(1),
            model_name: rs.getColumnValue(2),
            model_version: rs.getColumnValue(3),
            model_path: rs.getColumnValue(4),
            is_active: rs.getColumnValue(5),
            metrics: rs.getColumnValue(6),
            created_at: rs.getColumnValue(7)
        };
        return model;
    } else {
        return null;
    }
$$;

-- Create stored procedure to get model metrics
CREATE OR REPLACE PROCEDURE ML_MODELS.GET_MODEL_METRICS(MODEL_ID VARCHAR)
RETURNS VARIANT
LANGUAGE JAVASCRIPT
AS
$$
    var sql = `
        SELECT METRIC_NAME, METRIC_VALUE
        FROM ML_MODELS.MODEL_METRICS
        WHERE MODEL_ID = '${MODEL_ID}'
    `;
    
    var stmt = snowflake.createStatement({sqlText: sql});
    var rs = stmt.execute();
    
    var metrics = {};
    while (rs.next()) {
        var name = rs.getColumnValue(1);
        var value = rs.getColumnValue(2);
        metrics[name] = value;
    }
    
    return metrics;
$$;

-- Create role for the application
CREATE ROLE IF NOT EXISTS INSURANCE_PRICING_APP;

-- Grant privileges to the role
GRANT USAGE ON DATABASE INSURANCE_PRICING TO ROLE INSURANCE_PRICING_APP;
GRANT USAGE ON SCHEMA ML_MODELS TO ROLE INSURANCE_PRICING_APP;
GRANT USAGE ON SCHEMA PRICING TO ROLE INSURANCE_PRICING_APP;
GRANT USAGE ON SCHEMA STAGING TO ROLE INSURANCE_PRICING_APP;
GRANT USAGE ON WAREHOUSE INSURANCE_PRICING_WH TO ROLE INSURANCE_PRICING_APP;

GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA ML_MODELS TO ROLE INSURANCE_PRICING_APP;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA PRICING TO ROLE INSURANCE_PRICING_APP;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA STAGING TO ROLE INSURANCE_PRICING_APP;

GRANT USAGE ON ALL SEQUENCES IN SCHEMA ML_MODELS TO ROLE INSURANCE_PRICING_APP;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA PRICING TO ROLE INSURANCE_PRICING_APP;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA STAGING TO ROLE INSURANCE_PRICING_APP;

GRANT USAGE ON ALL PROCEDURES IN SCHEMA ML_MODELS TO ROLE INSURANCE_PRICING_APP;
GRANT USAGE ON ALL PROCEDURES IN SCHEMA PRICING TO ROLE INSURANCE_PRICING_APP;

-- Insert sample data for testing
-- Insert a sample model version
INSERT INTO ML_MODELS.MODEL_VERSIONS (
    MODEL_NAME, MODEL_VERSION, MODEL_PATH, IS_ACTIVE, METRICS
) VALUES (
    'decision_tree',
    '20250413000000',
    's3://insurance-pricing-models-dev/models/decision_tree_20250413000000.joblib',
    TRUE,
    PARSE_JSON('{"rmse": 120.5, "mae": 95.2, "r2": 0.85}')
);

-- Get the model ID
SET MODEL_ID = (SELECT MODEL_ID FROM ML_MODELS.MODEL_VERSIONS WHERE MODEL_VERSION = '20250413000000');

-- Insert sample feature importances
INSERT INTO ML_MODELS.FEATURE_IMPORTANCE (MODEL_ID, FEATURE_NAME, IMPORTANCE)
VALUES
    ($MODEL_ID, 'driver_age_mean', 0.25),
    ($MODEL_ID, 'vehicle_value_mean', 0.20),
    ($MODEL_ID, 'accident_count', 0.15),
    ($MODEL_ID, 'driver_experience_mean', 0.10),
    ($MODEL_ID, 'vehicle_age_mean', 0.08),
    ($MODEL_ID, 'location_crime_rate_mean', 0.07),
    ($MODEL_ID, 'policy_duration_days', 0.05),
    ($MODEL_ID, 'vehicle_per_driver_ratio', 0.04),
    ($MODEL_ID, 'years_since_last_incident', 0.03),
    ($MODEL_ID, 'incident_severity_score', 0.03);

-- Insert sample metrics
INSERT INTO ML_MODELS.MODEL_METRICS (MODEL_ID, METRIC_NAME, METRIC_VALUE)
VALUES
    ($MODEL_ID, 'rmse', 120.5),
    ($MODEL_ID, 'mae', 95.2),
    ($MODEL_ID, 'r2', 0.85);

-- Insert sample pricing history
INSERT INTO PRICING.PRICING_HISTORY (POLICY_ID, MODEL_ID, INPUT_FEATURES, BASE_PREMIUM, ADJUSTMENTS, FINAL_PREMIUM)
VALUES
    (
        UUID_STRING(),
        $MODEL_ID,
        PARSE_JSON('{"driver_age_mean": 35, "vehicle_value_mean": 25000, "accident_count": 0}'),
        1000.0,
        PARSE_JSON('{"credit_factor": 0.9, "territory_factor": 1.1}'),
        990.0
    ),
    (
        UUID_STRING(),
        $MODEL_ID,
        PARSE_JSON('{"driver_age_mean": 22, "vehicle_value_mean": 15000, "accident_count": 1}'),
        1200.0,
        PARSE_JSON('{"credit_factor": 1.2, "territory_factor": 1.0}'),
        1440.0
    ),
    (
        UUID_STRING(),
        $MODEL_ID,
        PARSE_JSON('{"driver_age_mean": 45, "vehicle_value_mean": 35000, "accident_count": 0}'),
        1100.0,
        PARSE_JSON('{"credit_factor": 0.8, "territory_factor": 0.9}'),
        792.0
    );

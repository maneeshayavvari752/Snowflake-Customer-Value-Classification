-- ============================================================
-- CUSTOMER VALUE CLASSIFICATION USING SNOWFLAKE ML
-- End-to-end assignment script
--
-- Run this script in a Snowflake SQL Worksheet.
-- If your role cannot CREATE DATABASE or CREATE WAREHOUSE,
-- ask your Snowflake administrator for the required privileges
-- or use an approved existing database/schema/warehouse.
--
-- Important:
-- 1. This script uses SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.
-- 2. The sample data is not a real churn dataset. This project
--    classifies customers by spending relative to the sample mean.
-- 3. The train/test split is deterministic and reproducible.
-- 4. CUSTOMER_ID is retained for reporting but excluded from
--    model features. TOTAL_SPENDING defines the label and is
--    intentionally excluded from model features to avoid leakage.
-- ============================================================

-- ============================================================
-- 1. ENVIRONMENT SETUP
-- ============================================================

CREATE DATABASE IF NOT EXISTS CUSTOMER_CHURN_DB;
CREATE SCHEMA IF NOT EXISTS CUSTOMER_CHURN_DB.ML_SCHEMA;

CREATE WAREHOUSE IF NOT EXISTS ML_WH
    WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

USE WAREHOUSE ML_WH;
USE DATABASE CUSTOMER_CHURN_DB;
USE SCHEMA ML_SCHEMA;

-- ============================================================
-- 2. EXPLORE SNOWFLAKE SAMPLE DATA
-- ============================================================

SHOW DATABASES;
SHOW SCHEMAS IN DATABASE SNOWFLAKE_SAMPLE_DATA;
SHOW TABLES IN SCHEMA SNOWFLAKE_SAMPLE_DATA.TPCH_SF1;

SELECT *
FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER
LIMIT 10;

SELECT *
FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS
LIMIT 10;

-- ============================================================
-- 3. CREATE CUSTOMER ML DATASET
-- ============================================================

CREATE OR REPLACE TABLE CUSTOMER_ML_DATA AS
WITH CUSTOMER_ORDER_STATS AS (
    SELECT
        C.C_CUSTKEY,
        C.C_ACCTBAL,
        C.C_MKTSEGMENT,
        COUNT(O.O_ORDERKEY) AS TOTAL_ORDERS,
        COALESCE(SUM(O.O_TOTALPRICE), 0) AS TOTAL_SPENDING,
        COALESCE(AVG(O.O_TOTALPRICE), 0) AS AVG_ORDER_VALUE
    FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER AS C
    LEFT JOIN SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS AS O
        ON C.C_CUSTKEY = O.O_CUSTKEY
    GROUP BY
        C.C_CUSTKEY,
        C.C_ACCTBAL,
        C.C_MKTSEGMENT
),
AVERAGE_SPENDING AS (
    SELECT AVG(TOTAL_SPENDING) AS AVG_SPENDING
    FROM CUSTOMER_ORDER_STATS
)
SELECT
    S.C_CUSTKEY AS CUSTOMER_ID,
    S.C_ACCTBAL,
    S.C_MKTSEGMENT,
    S.TOTAL_ORDERS,
    S.TOTAL_SPENDING,
    S.AVG_ORDER_VALUE,
    IFF(S.TOTAL_SPENDING > A.AVG_SPENDING, 1, 0) AS HIGH_VALUE
FROM CUSTOMER_ORDER_STATS AS S
CROSS JOIN AVERAGE_SPENDING AS A;

-- Dataset checks
SELECT * FROM CUSTOMER_ML_DATA LIMIT 10;

SELECT COUNT(*) AS TOTAL_CUSTOMERS
FROM CUSTOMER_ML_DATA;

SELECT HIGH_VALUE, COUNT(*) AS CUSTOMER_COUNT
FROM CUSTOMER_ML_DATA
GROUP BY HIGH_VALUE
ORDER BY HIGH_VALUE;

-- ============================================================
-- 4. REPRODUCIBLE TRAIN / TEST SPLIT
--    HASH-based split avoids separate RANDOM() results causing
--    train/test overlap or inconsistent reruns.
-- ============================================================

CREATE OR REPLACE TABLE CUSTOMER_TRAIN AS
SELECT *
FROM CUSTOMER_ML_DATA
WHERE MOD(ABS(HASH(CUSTOMER_ID)), 10) < 8;

CREATE OR REPLACE TABLE CUSTOMER_TEST AS
SELECT *
FROM CUSTOMER_ML_DATA
WHERE MOD(ABS(HASH(CUSTOMER_ID)), 10) >= 8;

-- Split counts
SELECT 'TRAIN' AS DATASET, COUNT(*) AS ROW_COUNT
FROM CUSTOMER_TRAIN
UNION ALL
SELECT 'TEST' AS DATASET, COUNT(*) AS ROW_COUNT
FROM CUSTOMER_TEST;

-- Class distribution
SELECT 'TRAIN' AS DATASET, HIGH_VALUE, COUNT(*) AS ROW_COUNT
FROM CUSTOMER_TRAIN
GROUP BY HIGH_VALUE
UNION ALL
SELECT 'TEST' AS DATASET, HIGH_VALUE, COUNT(*) AS ROW_COUNT
FROM CUSTOMER_TEST
GROUP BY HIGH_VALUE
ORDER BY DATASET, HIGH_VALUE;

-- ============================================================
-- 5. FEATURE ENGINEERING
--    Keep CUSTOMER_ID only for traceability. Do not feed it to
--    the model. TOTAL_SPENDING is excluded because it creates
--    HIGH_VALUE and would leak the answer to the model.
-- ============================================================

CREATE OR REPLACE TABLE CUSTOMER_TRAIN_FEATURES AS
SELECT
    CUSTOMER_ID,
    C_ACCTBAL,
    TOTAL_ORDERS,
    AVG_ORDER_VALUE,
    IFF(C_MKTSEGMENT = 'AUTOMOBILE', 1, 0) AS SEG_AUTOMOBILE,
    IFF(C_MKTSEGMENT = 'BUILDING', 1, 0) AS SEG_BUILDING,
    IFF(C_MKTSEGMENT = 'FURNITURE', 1, 0) AS SEG_FURNITURE,
    IFF(C_MKTSEGMENT = 'HOUSEHOLD', 1, 0) AS SEG_HOUSEHOLD,
    IFF(C_MKTSEGMENT = 'MACHINERY', 1, 0) AS SEG_MACHINERY,
    HIGH_VALUE
FROM CUSTOMER_TRAIN;

CREATE OR REPLACE TABLE CUSTOMER_TEST_FEATURES AS
SELECT
    CUSTOMER_ID,
    C_ACCTBAL,
    TOTAL_ORDERS,
    AVG_ORDER_VALUE,
    IFF(C_MKTSEGMENT = 'AUTOMOBILE', 1, 0) AS SEG_AUTOMOBILE,
    IFF(C_MKTSEGMENT = 'BUILDING', 1, 0) AS SEG_BUILDING,
    IFF(C_MKTSEGMENT = 'FURNITURE', 1, 0) AS SEG_FURNITURE,
    IFF(C_MKTSEGMENT = 'HOUSEHOLD', 1, 0) AS SEG_HOUSEHOLD,
    IFF(C_MKTSEGMENT = 'MACHINERY', 1, 0) AS SEG_MACHINERY,
    HIGH_VALUE
FROM CUSTOMER_TEST;

SELECT * FROM CUSTOMER_TRAIN_FEATURES LIMIT 10;
SELECT * FROM CUSTOMER_TEST_FEATURES LIMIT 10;

-- ============================================================
-- 6. PREPARE MODEL INPUT AND TRAIN CLASSIFICATION MODEL
--    CUSTOMER_ID is kept in the reporting feature table but
--    excluded from the actual model training columns.
-- ============================================================

CREATE OR REPLACE TABLE CUSTOMER_TRAIN_MODEL_INPUT AS
SELECT
    C_ACCTBAL,
    TOTAL_ORDERS,
    AVG_ORDER_VALUE,
    SEG_AUTOMOBILE,
    SEG_BUILDING,
    SEG_FURNITURE,
    SEG_HOUSEHOLD,
    SEG_MACHINERY,
    HIGH_VALUE
FROM CUSTOMER_TRAIN_FEATURES;

CREATE OR REPLACE SNOWFLAKE.ML.CLASSIFICATION CUSTOMER_VALUE_MODEL(
    INPUT_DATA => SYSTEM$REFERENCE('TABLE', 'CUSTOMER_TRAIN_MODEL_INPUT'),
    TARGET_COLNAME => 'HIGH_VALUE'
);

-- ============================================================
-- 7. GENERATE TEST PREDICTIONS
--    Pass predictor columns only; keep actual labels separately.
-- ============================================================

CREATE OR REPLACE TABLE CUSTOMER_TEST_SCORED AS
SELECT
    CUSTOMER_ID,
    HIGH_VALUE AS ACTUAL_VALUE,
    CUSTOMER_VALUE_MODEL!PREDICT(
        INPUT_DATA => {
            'C_ACCTBAL': C_ACCTBAL,
            'TOTAL_ORDERS': TOTAL_ORDERS,
            'AVG_ORDER_VALUE': AVG_ORDER_VALUE,
            'SEG_AUTOMOBILE': SEG_AUTOMOBILE,
            'SEG_BUILDING': SEG_BUILDING,
            'SEG_FURNITURE': SEG_FURNITURE,
            'SEG_HOUSEHOLD': SEG_HOUSEHOLD,
            'SEG_MACHINERY': SEG_MACHINERY
        }
    ) AS PREDICTION
FROM CUSTOMER_TEST_FEATURES;

SELECT *
FROM CUSTOMER_TEST_SCORED
LIMIT 10;

-- ============================================================
-- 8. CALCULATE ACCURACY
-- ============================================================

CREATE OR REPLACE TABLE CUSTOMER_EVALUATION AS
SELECT
    CUSTOMER_ID,
    ACTUAL_VALUE,
    PREDICTION:class::INT AS PREDICTED_VALUE,
    PREDICTION:probability:"1"::FLOAT AS HIGH_VALUE_PROBABILITY
FROM CUSTOMER_TEST_SCORED;

SELECT
    COUNT(*) AS TOTAL_TEST_ROWS,
    COUNT_IF(PREDICTED_VALUE = ACTUAL_VALUE) AS CORRECT_PREDICTIONS,
    ROUND(
        100.0 * COUNT_IF(PREDICTED_VALUE = ACTUAL_VALUE)
        / NULLIF(COUNT(*), 0),
        2
    ) AS ACCURACY_PERCENT
FROM CUSTOMER_EVALUATION;

-- ============================================================
-- 9. CONFUSION MATRIX
--    Rows = actual class; columns = predicted class.
-- ============================================================

SELECT
    ACTUAL_VALUE AS ACTUAL,
    PREDICTED_VALUE AS PREDICTED,
    COUNT(*) AS ROW_COUNT
FROM CUSTOMER_EVALUATION
WHERE PREDICTED_VALUE IS NOT NULL
GROUP BY ACTUAL_VALUE, PREDICTED_VALUE
ORDER BY ACTUAL_VALUE, PREDICTED_VALUE;

-- Optional pivot-style confusion matrix
SELECT
    ACTUAL_VALUE,
    COUNT_IF(PREDICTED_VALUE = 0) AS PREDICTED_0,
    COUNT_IF(PREDICTED_VALUE = 1) AS PREDICTED_1
FROM CUSTOMER_EVALUATION
GROUP BY ACTUAL_VALUE
ORDER BY ACTUAL_VALUE;

-- ============================================================
-- 10. PRECISION, RECALL, AND F1 SCORE
--    Calculate from this run's predictions; do not hard-code
--    confusion-matrix values.
-- ============================================================

WITH COUNTS AS (
    SELECT
        COUNT_IF(ACTUAL_VALUE = 0 AND PREDICTED_VALUE = 0) AS TN,
        COUNT_IF(ACTUAL_VALUE = 0 AND PREDICTED_VALUE = 1) AS FP,
        COUNT_IF(ACTUAL_VALUE = 1 AND PREDICTED_VALUE = 0) AS FN,
        COUNT_IF(ACTUAL_VALUE = 1 AND PREDICTED_VALUE = 1) AS TP
    FROM CUSTOMER_EVALUATION
)
SELECT
    TN,
    FP,
    FN,
    TP,
    ROUND(100.0 * TP / NULLIF(TP + FP, 0), 2) AS PRECISION_PERCENT,
    ROUND(100.0 * TP / NULLIF(TP + FN, 0), 2) AS RECALL_PERCENT,
    ROUND(
        100.0 * (2 * TP) / NULLIF((2 * TP) + FP + FN, 0),
        2
    ) AS F1_SCORE_PERCENT
FROM COUNTS;

-- ============================================================
-- 11. CREATE FINAL PREDICTION TABLE
-- ============================================================

CREATE OR REPLACE TABLE CUSTOMER_PREDICTIONS AS
SELECT
    CUSTOMER_ID,
    ACTUAL_VALUE,
    PREDICTED_VALUE,
    HIGH_VALUE_PROBABILITY
FROM CUSTOMER_EVALUATION;

SELECT *
FROM CUSTOMER_PREDICTIONS
LIMIT 10;

-- ============================================================
-- 12. FINAL PREDICTION SUMMARY AND VALIDATION
-- ============================================================

SELECT
    PREDICTED_VALUE,
    COUNT(*) AS CUSTOMER_COUNT
FROM CUSTOMER_PREDICTIONS
GROUP BY PREDICTED_VALUE
ORDER BY PREDICTED_VALUE;

SELECT
    COUNT(*) AS TOTAL_PREDICTIONS,
    COUNT_IF(PREDICTED_VALUE IS NULL) AS NULL_PREDICTIONS,
    COUNT_IF(ACTUAL_VALUE = PREDICTED_VALUE) AS CORRECT_PREDICTIONS,
    ROUND(
        100.0 * COUNT_IF(ACTUAL_VALUE = PREDICTED_VALUE)
        / NULLIF(COUNT(*), 0),
        2
    ) AS FINAL_ACCURACY_PERCENT
FROM CUSTOMER_PREDICTIONS;

-- Assignment objects created in CUSTOMER_CHURN_DB.ML_SCHEMA:
-- CUSTOMER_ML_DATA
-- CUSTOMER_TRAIN
-- CUSTOMER_TEST
-- CUSTOMER_TRAIN_FEATURES
-- CUSTOMER_TEST_FEATURES
-- CUSTOMER_TEST_SCORED
-- CUSTOMER_EVALUATION
-- CUSTOMER_PREDICTIONS
-- CUSTOMER_VALUE_MODEL

WITH charts AS (
    SELECT p.icustay_id, itemid, charttime, valuenum AS value, valueuom
    FROM `physionet-data.mimiciii_clinical.chartevents` p
    INNER JOIN `physionet-data.mimiciii_clinical.icustays` q ON p.icustay_id = q.icustay_id
    WHERE (itemid = 220615 -- serum creatinine
        OR itemid = 223835 -- FiO2
        OR itemid = 225668 -- lactic acid (lactate)
        OR itemid = 220644 -- ALT
        OR itemid = 220587 -- AST
        OR itemid IN (220050, 220179, 224167, 225309, 227243) -- systolic blood pressure    
        OR itemid IN (220051, 220180, 224643, 225310, 227242) -- diastolic blood pressure
        OR itemid IN (220052, 220181,         225312)         -- MAP
        OR itemid = 220224 -- PO2
        OR itemid = 220739 -- GCS eye opening
        OR itemid = 223900 -- GCS verbal response
        OR itemid = 223901 -- GCS motor response
        ) AND valuenum > 0
        AND error = 0
        AND DATETIME_DIFF(charttime, intime, SECOND)/3600 < 48
)
SELECT p.icustay_id, itemid, charttime, value, valueuom
FROM `physionet-275423.health_gym.ht_stays` p
INNER JOIN charts q ON p.icustay_id = q.icustay_id
ORDER BY icustay_id, itemid, charttime


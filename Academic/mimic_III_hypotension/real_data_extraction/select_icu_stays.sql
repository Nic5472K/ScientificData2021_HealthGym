WITH adult_admissions AS (
    -- admissions of adults (age≥18 years)
    SELECT
        hadm_id,
        MIN( DATETIME_DIFF(intime, dob, YEAR) ) AS first_admit_age,
        MIN (intime) AS first_icu_intime
    FROM `physionet-data.mimiciii_clinical.patients` p
    INNER JOIN `physionet-data.mimiciii_clinical.icustays` q ON p.subject_id = q.subject_id
    GROUP BY hadm_id
    HAVING first_admit_age >= 18
), first_stays AS (
    -- first ICU stays
    SELECT subject_id, p.hadm_id, icustay_id, intime
    FROM adult_admissions p
    LEFT JOIN `physionet-data.mimiciii_clinical.icustays` q ON (p.hadm_id = q.hadm_id AND p.first_icu_intime = q.intime)
    WHERE first_careunit = 'MICU' 
        AND dbsource = 'metavision'
        AND DATETIME_DIFF(outtime, intime, HOUR) >= 24
), low_maps AS (
    -- MAP measurements ≤65 mmHg measured within 48 hours
    SELECT p.icustay_id, itemid
    FROM `physionet-data.mimiciii_clinical.chartevents` p
    INNER JOIN `physionet-data.mimiciii_clinical.icustays` q ON p.icustay_id = q.icustay_id
    WHERE itemid IN (220052, 220181, 225312) --MAP items
        AND valuenum <= 65
        AND valuenum > 0
        AND error = 0
        AND DATETIME_DIFF(charttime, intime, HOUR) < 48
), ht_stays AS (
    -- ICU stays with acute hypotension (≥7 MAP ≤65mmHg)
    SELECT
        subject_id, hadm_id, p.icustay_id, intime,
        COUNT(itemid) AS num_low_maps
    FROM first_stays p
    LEFT JOIN low_maps q ON p.icustay_id = q.icustay_id
    GROUP BY subject_id, hadm_id, icustay_id, intime
    HAVING num_low_maps >= 7
)
SELECT subject_id, hadm_id, icustay_id, intime
FROM ht_stays
ORDER BY subject_id, hadm_id, icustay_id

WITH outputs AS (
    SELECT p.icustay_id, itemid, charttime, value, valueuom
    FROM `physionet-data.mimiciii_clinical.outputevents` p
    INNER JOIN `physionet-data.mimiciii_clinical.icustays` q ON p.icustay_id = q.icustay_id
    -- Relevant ITEMIDs, from https://github.com/vincentmajor/mimicfilters/blob/master/lists/OASIS_components/preprocess_urine_awk_str.txt
    -- Exluding 227489 (GU Irrigant/Urine Volume Out)
    WHERE itemid IN (226566, 226627, 226631, 226559, 226561, 226567, 226632, 226557, 226558, 226563) -- urine output
        AND value > 0
        AND iserror is null
        AND DATETIME_DIFF(charttime, intime, SECOND)/3600 < 48
)
SELECT p.icustay_id, itemid, charttime, value, valueuom
FROM `physionet-275423.health_gym.ht_stays` p
INNER JOIN outputs q ON p.icustay_id = q.icustay_id
ORDER BY icustay_id, itemid, charttime

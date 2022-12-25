WITH fluid_boluses AS (
    -- Fluid bolus therapy
    SELECT p.icustay_id, itemid, starttime, amount, amountuom
    FROM `physionet-data.mimiciii_clinical.inputevents_mv` p
    INNER JOIN `physionet-data.mimiciii_clinical.icustays` q ON p.icustay_id = q.icustay_id
    WHERE (itemid = 225158 -- NaCl 0.9% 
        OR itemid = 220955 -- Ringers lactate (empty)
        OR itemid = 225168 -- Packed red blood cells
        OR itemid = 220970 -- Fresh frozen plasma
        OR itemid = 225170 -- Platelets
        ) AND amount >= 250
        AND DATETIME_DIFF(endtime, starttime, MINUTE) < 60
        AND DATETIME_DIFF(starttime, intime, SECOND)/3600 < 48
)
SELECT p.icustay_id, itemid, starttime, amount, amountuom
FROM `physionet-275423.health_gym.ht_stays` p
INNER JOIN fluid_boluses q ON p.icustay_id = q.icustay_id
ORDER BY icustay_id, itemid, starttime

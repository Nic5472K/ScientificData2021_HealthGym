WITH vasopressors AS (
    -- Vasopressors
    SELECT p.icustay_id, itemid, starttime, endtime, rate, rateuom,
        CASE WHEN itemid = 221906 AND rateuom = 'mcg/kg/min' THEN round(cast(rate         as numeric),3) -- Norepinephrine
             WHEN itemid = 221906 AND rateuom = 'mcg/min'    THEN round(cast(rate/80      as numeric),3) -- Norepinephrine
             WHEN itemid = 222315 AND rate > 0.2             THEN round(cast(rate*5/60    as numeric),3) -- Vasopressin, in units/hour
             WHEN itemid = 222315 AND rateuom = 'units/hour' THEN round(cast(rate*5/60    as numeric),3) -- Vasopressin
             WHEN itemid = 222315 AND rateuom = 'units/min'  THEN round(cast(rate*5       as numeric),3) -- Vasopressin
             WHEN itemid = 221749 AND rateuom = 'mcg/kg/min' THEN round(cast(rate*0.45    as numeric),3) -- Phenylephrine
             WHEN itemid = 221749 AND rateuom = 'mcg/min'    THEN round(cast(rate*0.45/80 as numeric),3) -- Phenylephrine
             WHEN itemid = 221662 AND rateuom = 'mcg/kg/min' THEN round(cast(rate*0.01    as numeric),3) -- Dopamine
             WHEN itemid = 221662 AND rateuom = 'mcg/min'    THEN round(cast(rate*0.01/80 as numeric),3) -- Dopamine
             WHEN itemid = 221289 AND rateuom = 'mcg/kg/min' THEN round(cast(rate         as numeric),3) -- Epinephrine
             WHEN itemid = 221289 AND rateuom = 'mcg/min'    THEN round(cast(rate/80      as numeric),3) -- Epinephrine 
             ELSE null
        END AS rate_std
    FROM `physionet-data.mimiciii_clinical.inputevents_mv` p
    INNER JOIN `physionet-data.mimiciii_clinical.icustays` q ON p.icustay_id = q.icustay_id
    WHERE itemid IN (221906,222315,221749,221662,221289)
        AND rate > 0
        AND statusdescription <> 'Rewritten'
        AND DATETIME_DIFF(starttime, intime, SECOND)/3600 < 48
)
SELECT p.icustay_id, itemid, starttime, endtime, rate_std AS rate, rateuom
FROM `physionet-275423.health_gym.ht_stays` p
INNER JOIN vasopressors q ON p.icustay_id = q.icustay_id
ORDER BY icustay_id, itemid, starttime, endtime

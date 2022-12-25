# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Sebastiano Barbieri, UNSW.                                     +
#  All rights reserved. This file is part of the Health Gym, and is released under the   +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#  as part of this package.                                                              +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class Hyperparameters:
    """Hyperparameters"""

    # data
    mimic_dir = "../extracted_mimic_data/"
    data_dir = "../data/"

    # ids
    # charts
    serum_creatinine_ids = [220615]
    FiO2_ids = [223835]
    lactic_acid_ids = [225668]  # lactate
    ALT_ids = [220644]
    AST_ids = [220587]
    systolic_bp_ids = [220050, 220179, 224167, 225309, 227243]
    diastolic_bp_ids = [220051, 220180, 224643, 225310, 227242]
    MAP_ids = [220052, 220181, 225312]
    PO2_ids = [220224]
    GCS_eye_opening = [220739]
    GCS_verbal_response = [223900]
    GCS_motor_response = [223901]
    GCS_total_ids = [990000]  # 990000 is a custom id for total GCS
    # fluid boluses
    NaCl_09_ids = [225158]
    Ringers_lactate_ids = [220955]
    red_blood_cells_ids = [225168]
    fresh_frozen_plasma_ids = [220970]
    platelets_ids = [225170]
    # outputs
    urine_ids = [
        226566,
        226627,
        226631,
        226559,
        226561,
        226567,
        226632,
        226557,
        226558,
        226563,
    ]
    # vasopressors
    norepinephrine_ids = [221906]
    vasopressin_ids = [222315]
    phenylephrine_ids = [221749]
    dopamine_ids = [221662]
    epinephrine_ids = [221289]
    # aggregates
    bp_ids = systolic_bp_ids + diastolic_bp_ids + MAP_ids
    GCS_ids = GCS_eye_opening + GCS_verbal_response + GCS_motor_response
    fluid_boluses_ids = (
        NaCl_09_ids
        + Ringers_lactate_ids
        + red_blood_cells_ids
        + fresh_frozen_plasma_ids
        + platelets_ids
    )
    vasopressors_ids = (
        norepinephrine_ids
        + vasopressin_ids
        + phenylephrine_ids
        + dopamine_ids
        + epinephrine_ids
    )

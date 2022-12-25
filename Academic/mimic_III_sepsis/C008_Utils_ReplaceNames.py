###===>>>++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Copyright (c) 2021. by Nicholas Kuo & Sebastiano Babieri, UNSW.                     +
# All rights reserved. This file is part of the Health Gym, and is released under the +
# "MIT Lisence Agreement". Please see the LICENSE file that should have been included +
# as part of this package.                                                            +
###===###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

###===>>>
# This is the 8th of all files for WGAN on Sepsis

# The purpose is to add the unit of the variables to the end of their name
# We will be referring to Komorowski's description in 
# https://gitlab.doc.ic.ac.uk/AIClinician/AIClinician/-/blob/master/Dataset%20description%20Komorowski%20011118.xlsx

###===>>>
def Execute_C008(data_types):
    Replace_Names = [i for i in list(data_types["name"])]

    Replace_Names[0] = 'Age [days]'
    Replace_Names[1] = 'HR [bpm]'
    Replace_Names[2] = 'SysBP [mmHg]'
    Replace_Names[3] = 'MeanBP [mmHg]'
    Replace_Names[4] = 'DiaBP [mmHg]'

    Replace_Names[5] = 'RR [bpm]'
    Replace_Names[6] = 'SpO2 [%]'
    Replace_Names[7] = 'Temp [Celsius]'
    Replace_Names[8] = 'K [meq/L]'
    Replace_Names[9] = 'Na [meq/L]'

    Replace_Names[10] = 'Cl [meq/L]'
    Replace_Names[11] = 'Ca [mg/dL]'
    Replace_Names[12] = 'IonisedCa [mg/dL]'
    Replace_Names[13] = 'CO2 [meq/L]'
    Replace_Names[14] = 'Albumin [g/dL]'

    Replace_Names[15] = 'Hb [g/dL]'
    Replace_Names[16] = 'pH'
    Replace_Names[17] = 'BE [meq/L]'
    Replace_Names[18] = 'HCO3 [meq/L]'
    Replace_Names[19] = 'FiO2 [Fraction]'
    
    Replace_Names[20] = 'Glucose [mg/dL]'
    Replace_Names[21] = 'BUN [mg/dL]'
    Replace_Names[22] = 'Creatinine [mg/dL]'
    Replace_Names[23] = 'Mg [mg/dL]'
    Replace_Names[24] = 'SGOT [u/L]'
    
    Replace_Names[25] = 'SGPT [u/L]'
    Replace_Names[26] = 'TotalBili [mg/dL]'
    Replace_Names[27] = 'WbcCount [E9/L]'
    Replace_Names[28] = 'PlateletsCount [E9/L]'
    Replace_Names[29] = 'PaO2 [mmHg]'
    
    Replace_Names[30] = 'PaCO2 [mmHg]'
    Replace_Names[31] = 'Lactate [mmol/L]'
    Replace_Names[32] = 'InputTotal [mL]'
    Replace_Names[33] = 'Input4H [mL]'
    Replace_Names[34] = 'MaxVaso [mcg/kg/min]'
    
    Replace_Names[35] = 'OutputTotal [mL]'
    Replace_Names[36] = 'Output4H [mL]'
    Replace_Names[37] = 'Gender'
    Replace_Names[38] = 'ReAd'
    Replace_Names[39] = 'Mech'
    
    Replace_Names[40] = 'GCS'
    Replace_Names[41] = 'SpO2 [%]'
    Replace_Names[42] = 'PTT [s]'
    Replace_Names[43] = 'PT [s]'
    Replace_Names[44] = 'INR'

    return Replace_Names
    

    

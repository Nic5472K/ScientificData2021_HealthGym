
###===>>>
def Execute_C008(data_types):
    Replace_Names = [i for i in list(data_types["name"])]

    Replace_Names[0] = 'Age [days]'
    Replace_Names[1] = 'HR [bpm]'
    Replace_Names[2] = 'SysBP [mmHg]'
    Replace_Names[3] = 'MeanBP [mmHg]'
    Replace_Names[4] = 'DiaBP [mmHg]'

    Replace_Names[5] = 'RR [bpm]'
    Replace_Names[6] = 'Temp [Celsius]'
    Replace_Names[7] = 'K [meq/L]'
    Replace_Names[8] = 'Na [meq/L]'
    Replace_Names[9] = 'Cl [meq/L]'
    
    Replace_Names[10] = 'Ca [mg/dL]'
    Replace_Names[11] = 'IonisedCa [mg/dL]'
    Replace_Names[12] = 'CO2 [meq/L]'
    Replace_Names[13] = 'Albumin [g/dL]'
    Replace_Names[14] = 'Hb [g/dL]'
    
    Replace_Names[15] = 'pH'
    Replace_Names[16] = 'BE [meq/L]'
    Replace_Names[17] = 'HCO3 [meq/L]'
    Replace_Names[18] = 'FiO2 [Fraction]'
    Replace_Names[19] = 'Glucose [mg/dL]'
    
    Replace_Names[20] = 'BUN [mg/dL]'
    Replace_Names[21] = 'Creatinine [mg/dL]'
    Replace_Names[22] = 'Mg [mg/dL]'
    Replace_Names[23] = 'SGOT [u/L]'
    Replace_Names[24] = 'SGPT [u/L]'
    
    Replace_Names[25] = 'TotalBili [mg/dL]'
    Replace_Names[26] = 'WbcCount [E9/L]'
    Replace_Names[27] = 'PlateletsCount [E9/L]'
    Replace_Names[28] = 'PaO2 [mmHg]'
    Replace_Names[29] = 'PaCO2 [mmHg]'
    
    Replace_Names[30] = 'Lactate [mmol/L]'
    Replace_Names[31] = 'InputTotal [mL]'
    Replace_Names[32] = 'Input4H [mL]'
    Replace_Names[33] = 'MaxVaso [mcg/kg/min]'
    Replace_Names[34] = 'OutputTotal [mL]'
    
    Replace_Names[35] = 'Output4H [mL]'
    Replace_Names[36] = 'Gender'
    Replace_Names[37] = 'ReAd'
    Replace_Names[38] = 'Mech'
    Replace_Names[39] = 'GCS'
    
    Replace_Names[40] = 'SpO2 [%]'
    Replace_Names[41] = 'PTT [s]'
    Replace_Names[42] = 'PT [s]'
    Replace_Names[43] = 'INR'

    ###===>>>
    return Replace_Names
    

    

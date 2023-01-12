
###===>>>
def Execute_C008(data_types):
    Replace_Names = [i for i in list(data_types["name"])]
    
    #---
    Replace_Names[0] = 'Age [days]' 			# real begins
    Replace_Names[1] = 'HR [bpm]'
    Replace_Names[2] = 'SysBP [mmHg]'
    Replace_Names[3] = 'MeanBP [mmHg]'
    Replace_Names[4] = 'DiaBP [mmHg]'
    
    #---	
    Replace_Names[5] = 'RR [bpm]'
    Replace_Names[6] = 'K [meq/L]'
    Replace_Names[7] = 'Na [meq/L]'
    Replace_Names[8] = 'Cl [meq/L]'
    Replace_Names[9] = 'Ca [mg/dL]'

    #---
    Replace_Names[10] = 'IonisedCa [mg/dL]'
    Replace_Names[11] = 'CO2 [meq/L]'
    Replace_Names[12] = 'Albumin [g/dL]'
    Replace_Names[13] = 'Hb [g/dL]'    
    Replace_Names[14] = 'pH'
  
    #---   
    Replace_Names[15] = 'BE [meq/L]'
    Replace_Names[16] = 'HCO3 [meq/L]'
    Replace_Names[17] = 'FiO2 [Fraction]'
    Replace_Names[18] = 'Glucose [mg/dL]'
    Replace_Names[19] = 'BUN [mg/dL]'

    #---
    Replace_Names[20] = 'Creatinine [mg/dL]'
    Replace_Names[21] = 'Mg [mg/dL]'
    Replace_Names[22] = 'SGOT [u/L]'
    Replace_Names[23] = 'SGPT [u/L]'
    Replace_Names[24] = 'TotalBili [mg/dL]'

    #---
    Replace_Names[25] = 'WbcCount [E9/L]'
    Replace_Names[26] = 'PlateletsCount [E9/L]'
    Replace_Names[27] = 'PaO2 [mmHg]'
    Replace_Names[28] = 'PaCO2 [mmHg]'
    Replace_Names[29] = 'Lactate [mmol/L]'

    #---
    Replace_Names[30] = 'InputTotal [mL]'
    Replace_Names[31] = 'Input4H [mL]'
    Replace_Names[32] = 'MaxVaso [mcg/kg/min]'
    Replace_Names[33] = 'OutputTotal [mL]'
    Replace_Names[34] = 'Output4H [mL]'

    #---
    Replace_Names[35] = 'Gender' 				# bin begins
    Replace_Names[36] = 'ReAd'
    Replace_Names[37] = 'Mech'
    Replace_Names[38] = 'GCS'     				# cat begins
    Replace_Names[39] = 'SpO2 [%]'

    #---
    Replace_Names[40] = 'Temp [Celsius]'
    Replace_Names[41] = 'PTT [s]'
    Replace_Names[42] = 'PT [s]'
    Replace_Names[43] = 'INR'

    ###===>>>
    return Replace_Names
    

    

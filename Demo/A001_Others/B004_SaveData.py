import  numpy                   as      np
import  pandas                  as      pd
import  itertools
import  random
import  matplotlib.pyplot       as      plt
import  os

import  torch

from    A001_Others.B004zC002_BackTransform    import  *

###===>>>
def ExecuteB004(data_types, wgan_gp, Hyper002_Epochs):

    CurFolder = './Z001_Data/'+'Epoch_'+str(Hyper002_Epochs)+'/'
    if not os.path.exists(CurFolder):
        os.mkdir(CurFolder)

    torch_Fake = []
    with torch.no_grad():
        for itr in range(5):
            Generated_Len   = 20

            #---
            if itr < 4:
                temp = wgan_gp.generate_data(Generated_Len,
                                         500)
            else:
                temp = wgan_gp.generate_data(Generated_Len,
                                         164)

            ###===###
            Fake_Data = temp.view(-1, max(data_types["index_end"]))

            torch_Fake.append(Fake_Data)

    Fake_Data = torch.cat(torch_Fake, dim = 0)

    Fake_Data = Execute_C002(Fake_Data)

    Data = Fake_Data.clone()
    Data = Data.cpu().detach().numpy()
    tom = pd.DataFrame(Data)
    df_fake = pd.DataFrame()
    for index, row in data_types.iterrows():
        df_fake[row["name"]] = tom.iloc[:, index]
    
    #return Fake_Data, df_fake
    df_fake.to_csv(CurFolder+"Fake_Data.csv")

    ###===>>>
    return df_fake

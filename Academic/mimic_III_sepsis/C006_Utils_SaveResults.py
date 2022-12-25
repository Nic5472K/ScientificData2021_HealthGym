###===>>>++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Copyright (c) 2021. by Nicholas Kuo & Sebastiano Babieri, UNSW.                     +
# All rights reserved. This file is part of the Health Gym, and is released under the +
# "MIT Lisence Agreement". Please see the LICENSE file that should have been included +
# as part of this package.                                                            +
###===###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

###===>>>
import  numpy                       as      np
import  pandas                      as      pd
import  itertools
import  random
import  matplotlib.pyplot           as      plt
import  seaborn                     as      sns

import  torch

#---
from    C007_Utils_BackTransform    import  *
from    C008_Utils_ReplaceNames     import  *

###===>>>
# This is the 6th of all files for WGAN on Sepsis

###===>>>
# Below, we will compare our real, and fake data
def Execute_C006(wgan_gp,
                 data_types,
                 All_Trainable_Data
                 ):
    
    ###===>>>
    # (Part 1)      Data selection
    # In order to present our data easily,
    # we will randomly select a subset of the dataset for feature comparison
    Real_Data = All_Trainable_Data.view(-1, 96)
    
    Sample_Amount   = Real_Data.shape[0]//1000 * 1000
    Sampled_Points  = random.sample(range(Real_Data.shape[0]), Sample_Amount)
    Generated_Len   = 10

    #---
    Real_Data = All_Trainable_Data.view(-1, 96)
    Real_Data = Real_Data[Sampled_Points]

    temp = wgan_gp.generate_data(Generated_Len,
                                 int(Real_Data.shape[0]/Generated_Len))
    Fake_Data = temp.view(-1, 96)

    ###===>>>
    # (Part 2)      Feature back-transformation
    # And before we compare the data, we will need to
    # back-transform them to the state before re-scaled
    Real_Data = Execute_C007(Real_Data)
    Fake_Data = Execute_C007(Fake_Data)

    ###===>>>
    # (Part 3)      Variable comparison

    # We will also do a slight change to the names in data_type to add the units
    Replace_Names = Execute_C008(data_types)

    #---
    # (Part 3.1)    Float variables
    # Compared by overlaying distribution densities
    ncols = 4
    for itr in range(37):

        if (itr == 0) or (itr == 20):
            fig, ax = plt.subplots(ncols = ncols, nrows = 5, figsize = (45, 40))

        if itr < 20:
            i, j = itr // ncols, itr - (itr // ncols) * ncols
        else:
            itr2 = itr - 20
            i, j = itr2 // ncols, itr2 - (itr2 // ncols) * ncols
        
        #---
        CurName = Replace_Names[itr]

        #---
        Cur_Fake = Fake_Data[:, itr].view(-1).cpu().detach()
        Cur_Real = Real_Data[:, itr].view(-1).cpu()

        #---
        df_fake = pd.DataFrame()
        df_fake[CurName] = Cur_Fake
        df_fake["Type"]  = "Synthetic"
        df_real = pd.DataFrame()
        df_real[CurName] = Cur_Real
        df_real["Type"]  = "Real"
        df_all = pd.concat([df_fake, df_real], ignore_index = True)

        #---
        plot = sns.kdeplot(
            x = df_all[CurName].astype(float).values,
            hue = df_all["Type"],
            fill = True,
            ax = ax[i, j]
            )
        plot.legend_.set_title(None)
        ax[i, j].yaxis.set_visible(False)
        ax[i, j].set(xlabel = CurName)
        plt.tight_layout(pad = 3.0)

        if (itr == 19):
            plt.savefig('./SepsisImage/' + str(1).zfill(3) +
                        'distribution_comparison_floats_part01' +
                        '.png')
            plt.close()

        if (itr == 36):
            fig.delaxes(ax[4, 1])
            fig.delaxes(ax[4, 2])
            fig.delaxes(ax[4, 3])
            plt.savefig('./SepsisImage/' + str(1).zfill(3) +
                        'distribution_comparison_floats_part02' +
                        '.png')
            plt.close()
         
    #---
    # (Part 3.2)    nonFloat variables
    # Unlike the float variables which we
    # fully back-transformed in script C007
    # we require the quantile information for the categorical variables
    ReadFrom        = "./SepsisData/"
    BST_nonFloat    = torch.load(ReadFrom + "A001_BTS_nonFloat")

    ncols = 4
    fig, ax = plt.subplots(ncols = ncols, nrows = 2, figsize = (45, 16))
    # we will now compare the variable with side-by-side histograms
    for itr in range(37, Fake_Data.shape[1]):
        #---
        itr2 = itr - 37
        i, j = itr2 // ncols, itr2 - (itr2 // ncols) * ncols

        #---
        CurName = Replace_Names[itr]
        
        Cur_Fake = Fake_Data[:, itr].long().view(-1).cpu().detach().numpy()
        Cur_Real = Real_Data[:, itr].long().view(-1).cpu().numpy()

        #---
        df_fake = pd.DataFrame()
        df_fake[CurName] = Cur_Fake
        df_fake["Type"]  = "Synthetic"
        df_real = pd.DataFrame()
        df_real[CurName] = Cur_Real
        df_real["Type"]  = "Real"

        #---
        # below, we set the xticks for the categorical values
        # we first consider the binary variables
        if itr <= 39:

            if itr == 37:
                # for gender, Komorosski has stated that
                # 0 is for male and that 1 is for female
                # in his data description in
                # https://gitlab.doc.ic.ac.uk/AIClinician/AIClinician/-/blob/master/Dataset%20description%20Komorowski%20011118.xlsx
                mapper = {"0":"Male", "1":"Female"}
            else:
                # otherwise, it is just simply False vs True
                mapper = {"0":"False", "1":"True"}

        elif itr == 40:
            # and this one here is GCS
            mapper = {str(itr3):itr3+3 for itr3 in range(0, 13)}


        # and below is for every variables that are CAT_NLN
        else:
            # we read the previously saved quantile statistics
            # recorded from script A001
            CurQuantiles = BST_nonFloat["Quantiles"][itr - 37]
            mapper = {str(itr3): "<" + '{:.2f}'.format(CurQuantiles[itr3 + 1]) for itr3 in range(1, 9)}
            mapper[str(0)] = '{:.2f}'.format(CurQuantiles[0]) + "< x <" + '{:.2f}'.format(CurQuantiles[1])
            mapper[str(9)] = "<=" + '{:.2f}'.format(CurQuantiles[10])
    
        #---
        df_all = pd.concat([df_fake, df_real], ignore_index = True)

        df_all[CurName] = df_all[CurName].\
                          astype(str).\
                          astype("category").\
                          map(mapper)

        #---
        plot = sns.histplot(
                x = df_all[CurName],
                hue = df_all["Type"],
                fill = True,
                multiple = "dodge",
                stat = "probability",
                shrink = 0.8,
                alpha = 0.3,
                linewidth = 0,
                ax = ax[i, j]
                )
        # the names of the xtick will be a bit long for CAT_NLN
        # so we will rotate the ticks
        # and we will move the first class slightly to the left
        if itr > 40:
            ax[i,j].tick_params(labelrotation=30)
            temp = ax[i, j].get_xticks()
            temp[0] = -0.5
            ax[i, j].set_xticks(temp)

        plot.legend_.set_title(None)
        ax[i, j].yaxis.set_visible(False)
        ax[i, j].set(xlabel = CurName)

    plt.savefig('./SepsisImage/' + str(2).zfill(3) +
                'distribution_comparison_nonfloat' +
                '.png')
    plt.close()

    #---
    # (Part 3.3)        Correlation comparison
    # Individual variables may behave nicely separately
    # but we want to test and see if the WGAN captured
    # the inherent relationships within the data variables themselves

    # We will need to put the dataset into pandas
    # in order to compute the correlations

    # We will change the name slightly,
    # we no longer need the Demo002 coding any more
    name_replace = [i[8:] for i in list(data_types["name"])]
    DF_Real = pd.DataFrame(data = Real_Data.detach().numpy(),
                           columns = name_replace)
    DF_Fake = pd.DataFrame(data = Fake_Data.cpu().detach().numpy(),
                           columns = name_replace)

    real_matrix = DF_Real.astype(float).corr()
    fake_matrix = DF_Fake.astype(float).corr()

    mask = np.triu(np.ones_like(real_matrix, dtype = bool))

    fig, ax = plt.subplots(ncols = 1, nrows = 2, figsize = (25, 40))
    with sns.axes_style("white"):
        for i, (matrix, data_type) in enumerate(
            zip([fake_matrix, real_matrix], ["Synthetic", "Real"])
            ):
            sns.heatmap(
                matrix,
                cmap = "coolwarm",
                mask = mask,
                vmin = "-1",
                vmax = "1",
                linewidths = 0.5,
                square = True,
                ax = ax[i],
                ).set_title("Correlation Matrix: " + data_type + "Data")

    fig.tight_layout(pad = 3.0)

    plt.savefig('./SepsisImage/' + str(3).zfill(3) +
                'correlation_comparison' +
                '.png')
    plt.close()
        













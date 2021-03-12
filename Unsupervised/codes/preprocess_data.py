import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data():
    names = ['mad','rsc','gra','ms','ngr']
    df = []
    for name in names:
        df.append(pd.read_csv("./data_raw/%s_13_1_2021.csv"%name))

    feature_names = [['Depth CSF-B (m)','Moisture dry (wt%)','Moisture wet (wt%)','Bulk density (g/cm³)','Dry density (g/cm³)','Grain density (g/cm³)','Porosity (vol%)'],
                    ['Depth CSF-B (m)','Reflectance L*','Reflectance a*','Reflectance b*','Tristimulus X','Tristimulus Y','Tristimulus Z'],
                    ['Depth CSF-B (m)','Bulk density (GRA)'],
                    ['Depth CSF-B (m)','Magnetic susceptibility (instr. units)'],
                    ['Depth CSF-B (m)','NGR total counts (cps)']
                    ]
    df_sort = []
    for i in range(5):
        #print(df[i].shape)
        d_temp = df[i][feature_names[i]]
        d_temp = d_temp.sort_values('Depth CSF-B (m)')
        d_temp.index = range(len(d_temp))
        d_temp = d_temp.dropna(axis=0,how='any')
        #print(d_temp.shape)
        d_temp.to_csv("./data/%s.csv"%names[i])
        df_sort.append(d_temp)

    df1 = df_sort[0]
    for i in range(1,5):
        df1 = pd.merge_asof(left = df1, right = df_sort[i],on = 'Depth CSF-B (m)',direction = "nearest")
    #print(df1.shape)
    pd.options.mode.chained_assignment = None 
    df1 = df1.drop(index=[19,20,21,22,23,358,359,360,363,364,365,368])
    df1.index = range(len(df1))
    df1.to_csv("./data/data_merged.csv")
    return df1
    



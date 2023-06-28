
import pandas as pd
import os
import shutil
from tqdm import tqdm

def xlsx_to_csv(datafile):
    if os.path.isfile(datafile):
        df = pd.read_excel(datafile, index_col=[0])
        df.to_csv("./coordinates_Bbox.csv", sep=",")
    


def generate_dataset(copy_files=False):
    xls = pd.ExcelFile('coordinate in Bbox_Sept24.xlsx')
    df1 = pd.read_excel(xls, 'CA')
    df2 = pd.read_excel(xls, 'WFBB_NE')
    df3 = pd.read_excel(xls, 'IL')
    df4 = pd.read_excel(xls, 'ND')

    #adjust table id's to go in increasing order across all tables
    max_id1 = df1['Sample ID'].max()
    df2['Sample ID'] += max_id1

    max_id2 = df2['Sample ID'].max()
    df3['Sample ID'] += max_id2

    max_id3 = df3['Sample ID'].max()
    df4['Sample ID'] += max_id3

    #Do the same for culvert id's to keep them unique
    max_id1 = df1['CulvertID'].max()
    df2['CulvertID'] += max_id1

    max_id2 = df2['CulvertID'].max()
    df3['CulvertID'] += max_id2

    max_id3 = df3['CulvertID'].max()
    df4['CulvertID'] += max_id3

    print(df1)
    print(df2)
    print(df3)
    print(df4)

    combined_df = pd.concat([df1, df2], ignore_index=True, axis=0)
    combined_df = pd.concat([combined_df, df3], ignore_index=True, axis=0)
    combined_df = pd.concat([combined_df, df4], ignore_index=True, axis=0)

    print(combined_df)

    #Save new xlsx file
    combined_df.to_excel('compined_set.xlsx')

    #Generate new folder, copying files from each directory with their new name

    if(copy_files):
        OUT_PATH = './combined_data'
        P1 = './CA'
        P2 = './WFBB_NE'
        P3 = './IL'
        P4 = './ND'

        path_list = [P1,P2,P3,P4]

        if not os.path.exists(OUT_PATH):
            os.makedirs(OUT_PATH)

        file_number = 1
        for path in path_list:
            file_list = os.listdir(path)
            for filename in tqdm(file_list):
                file_path = os.path.join(path,filename)
                print(file_path)
                shutil.copy(file_path, os.path.join(OUT_PATH,str(file_number)+'.tif'))
                file_number += 1



generate_dataset()


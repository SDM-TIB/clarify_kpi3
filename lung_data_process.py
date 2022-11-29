import os
import pandas as pd
import numpy as np

def all_mutation():
    
    def mut_type(fn):
        l_ = ''
        if 'alk' in fn:
            l_ = 'ALK'
        elif 'egfr' in fn:
            l_ = 'EGFR'
        elif 'her2' in fn:
            l_ = 'HER2'
        elif 'pdl1' in fn:
            l_ = 'PDL1'
        elif 'ros1' in fn:
            l_ = 'ROS1'
        elif 'biomarker_mutation.csv' in fn:
            l_='Other'
        else:
            return None, None
        
        flag = 1 if 'positive' in fn else -1 if 'negative' in fn else 0

        return l_, flag

    data_ = {"Patient_id":[], "Mutation":[]}
    dir_name = 'positive_negative_mutation'
    for fname in os.listdir(dir_name):
        fn = os.path.join(dir_name, fname)
        mutation, pos_neg = mut_type(fname)
        if mutation is None:
            continue
        
        if 'Other' == mutation:
            df = pd.read_csv(fn)
            pos_id = set(pd.read_csv(dir_name+'/biomarker_mutation_positive.csv').EHR)
            df1 = df.loc[(df.EHR.isin(pos_id))]
            for i in range(len(df1)):
                data_['Patient_id'].append(df1.iloc[i,0])
                data_['Mutation'].append(df1.iloc[i, 2])
            df2 = df.loc[(~df.EHR.isin(pos_id))]
            for i in range(len(df2)):
                data_['Patient_id'].append(df2.iloc[i,0])
                data_['Mutation'].append('NoMutation')
            continue
        
        df = pd.read_csv(fn)
        for i in range(len(df)):
            p_id = str(df.iloc[i, 0])
            p_id = p_id.replace('http://clarify2020.eu/entity/', '')
            data_['Patient_id'].append(p_id)
            data_['Mutation'].append(mutation if pos_neg == 1 else "NoMutation" if pos_neg == -1 else "NULL")

    
    df = pd.DataFrame(data_).drop_duplicates()

    # remove those records of UNK patient who already have with either positive result or negative result
    p_ids = set(df.loc[(df.Mutation != 'NULL')].Patient_id)
    df = df.loc[~((df.Patient_id.isin(p_ids) ) & (df.Mutation == 'NULL'))]
    # remove those NoMutation record if patient has any positive mutation
    p_ids = set(df.loc[(df.Mutation != 'NULL') & (df.Mutation != 'NoMutation')].Patient_id)
    df = df.loc[~((df.Patient_id.isin(p_ids)) & (df.Mutation == 'NoMutation'))]

    df.sort_values('Mutation').to_csv('patient_mutation.csv', index=False)

            



all_mutation()



def no_mutations():
    dir_name = 'positive_negative_mutation'
    pos_pids = set()
    neg_pids = set()
    for fname in os.listdir(dir_name):
        pids = set([str(e).strip() for e in pd.read_csv(os.path.join(dir_name, fname))['EHR'].values])
        if 'negative' in fname:
            # print('neg', fname)
            neg_pids = neg_pids.union(pids)
        elif 'positive' in fname:
            # print('pos', fname)
            pos_pids = pos_pids.union(pids)

    no_mutation_pids = neg_pids.difference(pos_pids)
    return no_mutation_pids

def family_number_diversity():
    return pd.read_csv('data4kpi3/family_num.csv')
    

def patient_age():
    return pd.read_csv('data4kpi3/patient_age.csv')


def to_degree(f_):
        if '1' in f_:
            return '1st'
        elif '2' in f_:
            return '2nd'
        elif '3' in f_:
            return '3rd'
        else:
            return f_


def to_gender(f_):
    if 'M' in f_:
        return "M"
    elif 'F' in f_:
        return 'F'
    else:
        return f_


def clean_df(df):
    # Rename columns
    original_feature_names = ["familyNum","familyDiversity", "EHR","AgeCategory","Gender", "Smoking_habits", "Stages","Drug","familyType","CancerType","Biomarker","Relapse", "Date", "HasFamily"]
    new_feature_names = ["FamilyNum","FamilyDiversity", "Patient_id", 'Age', "Gender", "Smoker", "Stage", "Treatment", "Family", "FamilyCancer", "Mutation", "Relapse", "Date", "HasFamily"]
    df.rename(columns = dict(zip(original_feature_names, new_feature_names)), inplace = True)

    df1 = family_number_diversity()
    df1.rename(columns = dict(zip(original_feature_names, new_feature_names)), inplace = True)
    df1['Patient_id'] = df1['Patient_id'].str.replace("http://clarify2020.eu/entity/", '', regex=False)

    # remove prefix url
    df.replace('\s+','',regex=True,inplace=True) 
    for col in df.columns:
        if df[col].dtype != np.float64:
            df[col] = df[col].str.replace("http://clarify2020.eu/entity/", '', regex=False)
    
    # Add Family Number and Family Diversity
    print("intersection ---- df : df1 ", len(set(df1.Patient_id).intersection(set(df.Patient_id))))
    df = df.merge(df1, on='Patient_id', how='left')

    # replace mutation clean by considering 'No Mutation"
    no_mutation_id = no_mutations()
    df['Mutation'] = df['Mutation'].mask(df.Patient_id.isin(no_mutation_id), 'NoMutation')

    # add family gender and family degree
    family = ["UNK", "Father", "Mother", "Brother", "Sister", "Daughter", "Son", "Uncle", "Nephew", "Grandfather", "Grandmother", "Aunt", "Niece", 'Granddaughter', 'Grandson', 'Grandgrandfather', 'Grandgrandmother', "No", 'Halfsister', 'Halfbrother', 'Female_Cousin', 'Male_Cousin', 'NULL']
    family_replace = ["UNK", "M1", 'F1', 'M1', 'F1', 'F1', 'M1', 'M2', 'M2', 'M2', 'F2', 'F2', 'F2', 'F2', 'M2', 'M3', 'F3', 'No', 'F2', 'M2', 'F3', 'M3', 'NULL']
    
    df['FamilyGender'] = df['Family'].replace(to_replace={f: to_gender(family_replace[i]) for i, f in enumerate(family)})
    df["FamilyDegree"] = df['Family'].replace(to_replace={f: to_degree(family_replace[i]) for i, f in enumerate(family)})
    
    # replace columns related to family
    def process_family_col(col_n):
        df[col_n] = df[col_n].mask(df.HasFamily == 'No', 'No')
        df[col_n] = df[col_n].mask(df.HasFamily == 'UNK', 'UNK')
    for col in df.columns:
        if 'HasFamily' == col:
            continue
        if 'Family' in col or 'family' in col:
            process_family_col(col)

    # Update age
    df2 = patient_age()
    df2.rename(columns = dict(zip(original_feature_names, new_feature_names)), inplace = True)

    df2['Patient_id'] = df2['Patient_id'].replace('"', '', regex=False)
    
    df2['Age'] = df2['Age'].replace('"', '', regex=False)
    df.drop(columns=['Age'], inplace=True)
    
    df['Patient_id'] = df['Patient_id'].astype(np.int64)
    print("intersection ---- df : df2 ", len(set(df2.Patient_id).intersection(set(df.Patient_id))))
    df = df.merge(df2, on='Patient_id', how='left')
    
    df = df.fillna('NULL')
    return df


def unit_table():
    univeral = pd.read_csv('data4kpi3/clean_df.csv')



################################################################################
##              Step 1. Get Universial Table clean.df
################################################################################

# df = pd.read_csv('sparql6.csv')
# df = clean_df(df)
# df.to_csv('data4kpi3/clean_df.csv', index=False)


################################################################################
##              Step 2. Get Unit Table 
################################################################################



import pandas as pd
import numpy as np
import json
import re
from nltk.corpus import stopwords

def convert_json_to_csv(json_file):
    data = pd.DataFrame.from_dict(json_file, orient='columns')
    data.to_csv('CAD_test.csv', index=False)
    
    return data

def preprocess():
    with open('CAD_test.csv', encoding="utf-8") as h:
        data = pd.read_csv(h)

    data = data.rename(columns={"Indication": "clinical_diagnosis", 
                                "Tech": "techniques", 
                                "Conclusion": "conclusion", 
                                "性別":"sex",
                                "AGE":'realage',
                               "DATE":'operation day',
                               "身份證號":'id'})
    # data.drop(data.columns.difference(['ID',"clinical_diagnosis","techniques", "conclusion",'D5Y', 'realage', 'sex','operation day']), 1, inplace=True)
    # data = data.rename(columns={"Reports": "TEXT", "D5Y": "Label"})

    data['realage'] = data['realage'].fillna(65.56)
    data['realage'] = data['realage'].astype(int)
    data['sex'] = data['sex'].str.replace('M', 'male')
    data['sex'] = data['sex'].str.replace('F', 'female')
    data['realage'] = data['realage'].astype(str)
    data['operation day'] = data['operation day'].astype(str)
    data['sex'] = data['sex'].astype(str)
    data['clinical_diagnosis'] = 'age is ' + data['realage'].astype(str) + ' gender is '+ data['sex'].astype(str)+ ' operation day is ' + data['operation day'].astype(str) + " "+ data['clinical_diagnosis'].astype(str)
    data['clinical_diagnosis_conclusion'] = data['clinical_diagnosis']+data['conclusion']
    data['techniques'] = 'age is ' + data['realage'].astype(str) + ' gender is '+ data['sex'].astype(str)+ ' operation day is ' + data['operation day'].astype(str) + " "+ data['techniques'].astype(str)
    data['conclusion'] = 'age is ' + data['realage'].astype(str) + ' gender is '+ data['sex'].astype(str)+ ' operation day is ' + data['operation day'].astype(str) + " "+ data['conclusion'].astype(str)

    def preprocessing_column(column):

        data['TEXT'] = data[column]
        data['TEXT'] = data['TEXT'].astype(str)


        data['TEXT'] = data['TEXT'].str.replace('CLINICAL DIAGNOSIS :', '')
        data['TEXT'] = data['TEXT'].str.replace('CLINICAL DIAGNOSIS:', '')
        data['TEXT'] = data['TEXT'].str.replace('STEMI', 'st elevation myocardial infarction')
        data['TEXT'] = data['TEXT'].str.replace('PCI', 'Percutaneous Coronary Intervention')
        data['TEXT'] = data['TEXT'].str.replace('VT', 'Ventricular tachycardia')
        data['TEXT'] = data['TEXT'].str.replace('CAD-DVD', 'Coronary artery disease-double vessel disease')
        data['TEXT'] = data['TEXT'].str.replace('CAD, DVD', 'Coronary artery disease-double vessel disease')
        data['TEXT'] = data['TEXT'].str.replace('CAD', 'Coronary artery disease')
        data['TEXT'] = data['TEXT'].str.replace('LM lesion', 'left main lesion')
        data['TEXT'] = data['TEXT'].str.replace('CABG', 'Coronary artery bypass graft')
        data['TEXT'] = data['TEXT'].str.replace('CTO', 'chronic total occlusion')
        data['TEXT'] = data['TEXT'].str.replace('AS change', 'atherosclerotic change')
        data['TEXT'] = data['TEXT'].str.replace('ACLS', 'advanced Cardiac Life Support')
        data['TEXT'] = data['TEXT'].str.replace('CPR', 'Cardiopulmonary Resuscitation')
        data['TEXT'] = data['TEXT'].str.replace('rfv ', 'right femory vein ')
        data['TEXT'] = data['TEXT'].str.replace('amp ', 'ample ')
        data['TEXT'] = data['TEXT'].str.replace('ao ', 'aorta ')
        data['TEXT'] = data['TEXT'].str.replace('r\nt ', 'right ')
        data['TEXT'] = data['TEXT'].str.replace('ra ', 'right atrium ')
        data['TEXT'] = data['TEXT'].str.replace('rv ', 'right ventricle ')
        data['TEXT'] = data['TEXT'].str.replace('pcw ', 'pulmonary pressure ')
        data['TEXT'] = data['TEXT'].str.replace('lad ', 'left artery disease ')
        data['TEXT'] = data['TEXT'].str.replace('pa ', 'pulmonary artery ')
        data['TEXT'] = data['TEXT'].str.replace('ia ', 'intra artery ')
        data['TEXT'] = data['TEXT'].str.replace('lm ', 'left main ')
        data['TEXT'] = data['TEXT'].str.replace('mdct', 'Computed Tomography ')
        data['TEXT'] = data['TEXT'].str.replace('diagnosi ', 'diagnosis ')
        data['TEXT'] = data['TEXT'].str.replace('MEDICATION ', '')
        data['TEXT'] = data['TEXT'].str.replace('TECHNIQUES ', '')
        data['TEXT'] = data['TEXT'].str.replace('2. PREMEDICATION:,', '')
        data['TEXT'] = data['TEXT'].str.replace('Allermine 1 amp,', '')
        data['TEXT'] = data['TEXT'].str.replace('3. TECHNIQUES:', '')

        # remove stop words
        #import nltk
        #nltk.download('stopwords')
        
        stop = stopwords.words('english')
        data['TEXT'].apply(lambda x: [item for item in x if item not in stop])


        def clean_text(x):
            x = " ".join(x.split())
            x= " ".join((" ".join(x.split("[**"))).split("**]"))
            x = re.sub(r"\([^()]*\)", "", x)
            key_value_strip =(x.split(":"))
            ##remove all sub strings which have a length lesser than 50 characters
            string = " ".join([sub_unit for sub_unit in key_value_strip if len(sub_unit)>3])
            x = re.sub(r"(\d+)+(\.|\))", "", string)## remove all serialization eg 1. 1)
            x = re.sub(r"(\*|\?|=)+", "", x) ##removing all *, ? and =
            x = re.sub(r"\b(\w+)( \1\b)+", r"\1", x) ## removing consecutive dupicate words
            x = x.replace("FOLLOW UP", "FOLLOWUP")
            x = x.replace("FOLLOW-UP", "FOLLOWUP")
            x = x.replace("*", '')
            x = re.sub(r"(\b)(f|F)(irst)(\b)?[\d\-\d]*(\s)*(\b)?(n|N)(ame)[\d\-\d]*(\s)*[\d\-\d]*(\b)","",x)##remove firstname
            x = re.sub(r"(\b)(l|L)(ast)(\b)?[\d\-\d]*(\s)*(\b)?(n|N)(ame)[\d\-\d]*(\s)*[\d\-\d]*(\b)", "", x)
            x = re.sub(r"(\b)(d|D)\.?(r|R)\.?(\b)", "", x) #remove Dr abreviation
            x = re.sub(r"([^A-Za-z0-9\s](\s)){2,}", "", x)##remove consecutive punctuations
            return(x.replace("  ", " "))


        data['TEXT'] = data['TEXT'].apply(lambda x: clean_text(x))
        data[column] = data['TEXT']
    preprocessing_column('clinical_diagnosis_conclusion')
    preprocessing_column('techniques')

    data.drop(data.columns.difference(['id','clinical_diagnosis_conclusion','Label', 'operation day', 'techniques']), 1, inplace=True)
    # data['Label'] = data['Label'].fillna(0)
    # data = data[:10]


    data.to_csv('CAD_test10_preprocessed.csv', index=False)
    return "preprocessed!"
# pip install pycaret
import pandas as pd
import jinja2
from datasets import load_dataset

from pycaret.classification import *
import imblearn as im
import sklearn

import gradio as gr
import numpy as np

import io
import pickle
import requests
import urllib.request
import shutil

url = 'https://raw.githubusercontent.com/fmegahed/tavr_paper/main/data/example_data2.csv'
download = requests.get(url).content

ex_data =pd.read_csv(io.StringIO(download.decode('utf-8')))
ex_data = ex_data.to_numpy()
ex_data = ex_data.tolist()


def predict(age, female, race, elective, aweekend, zipinc_qrtl, hosp_region, hosp_division, hosp_locteach,
            hosp_bedsize, h_contrl, pay, anemia, atrial_fibrillation, 
            cancer, cardiac_arrhythmias, carotid_artery_disease, 
            chronic_kidney_disease, chronic_pulmonary_disease, coagulopathy,
            depression, diabetes_mellitus, drug_abuse, dyslipidemia, endocarditis,
            family_history, fluid_and_electrolyte_disorder, heart_failure,
            hypertension, known_cad, liver_disease, obesity, peripheral_vascular_disease,
            prior_cabg, prior_icd, prior_mi, prior_pci, prior_ppm, prior_tia_stroke,
            pulmonary_circulation_disorder, smoker, valvular_disease, weight_loss,
            endovascular_tavr, transapical_tavr):

  df = pd.DataFrame.from_dict({
      'age': [age], 'female': [female], 'race': [race], 'elective': elective,
       'aweekend': [aweekend], 'zipinc_qrtl': [zipinc_qrtl], 
       'hosp_region': [hosp_region], 'hosp_division': [hosp_division],
       'hosp_locteach': [hosp_locteach], 'hosp_bedsize': [hosp_bedsize],
       'h_contrl': [h_contrl], 'pay': [pay], 'anemia': [anemia], 
       'atrial_fibrillation': [atrial_fibrillation], 'cancer': [cancer],
       'cardiac_arrhythmias': [cardiac_arrhythmias], 
       'carotid_artery_disease': [carotid_artery_disease], 
       'chronic_kidney_disease': [chronic_kidney_disease], 
       'chronic_pulmonary_disease': [chronic_pulmonary_disease], 
       'coagulopathy': [coagulopathy], 'depression': [depression],
       'diabetes_mellitus': [diabetes_mellitus], 'drug_abuse': [drug_abuse], 
       'dyslipidemia': [dyslipidemia], 'endocarditis': [endocarditis],
       'family_history': [family_history], 'fluid_and_electrolyte_disorder': [fluid_and_electrolyte_disorder],
       'heart_failure': [heart_failure], 'hypertension': [hypertension],
       'known_cad': [known_cad], 'liver_disease': [liver_disease],
       'obesity': [obesity], 'peripheral_vascular_disease': [peripheral_vascular_disease],
       'prior_cabg': [prior_cabg], 'prior_icd': [prior_icd], 'prior_mi': [prior_mi],
       'prior_pci': [prior_pci], 'prior_ppm': [prior_ppm], 'prior_tia_stroke': [prior_tia_stroke],
       'pulmonary_circulation_disorder': [pulmonary_circulation_disorder], 
       'smoker': [smoker], 'valvular_disease': [valvular_disease],
       'weight_loss': [weight_loss], 'endovascular_tavr': [endovascular_tavr],
       'transapical_tavr': [transapical_tavr]
  })
  
  df.loc[:, df.dtypes == 'object'] =\
    df.select_dtypes(['object'])\
    .apply(lambda x: x.astype('category'))

  # converting ordinal column to ordinal
  df.zipinc_qrtl = df.zipinc_qrtl.astype(ordinal_cat)
  
  # reading the model from GitHub
  with urllib.request.urlopen('https://github.com/fmegahed/tavr_paper/blob/main/data/final_model.pkl?raw=true') as response, open('final_model.pkl', 'wb') as out_file:
    shutil.copyfileobj(response, out_file)

  model = load_model('final_model')
  
  
  pred = predict_model(model, df, raw_score=True)
  
  return {'Death %': round(100*pred['Score_Yes'][0], 2),
       'Survival %': round(100*pred['Score_No'][0], 2),
       'Predicting Death Outcome:': pred['Label'][0]}

# Defining the containers for each input
age = gr.inputs.Slider(minimum=0, maximum=100, default=60, label="Age")
female = gr.inputs.Dropdown(choices=["Female", "Male"],label = 'Sex')
race = gr.inputs.Dropdown(choices=['Asian or Pacific Islander', 'Black', 'Hispanic', 'Native American', 'White',  'Other'], label = 'Race')
elective = gr.inputs.Radio(choices=['Elective', 'NonElective'], label = 'Elective')
aweekend = gr.inputs.Radio(choices=["No", "Yes"], label = 'Weekend')
zipinc_qrtl = gr.inputs.Radio(choices=['FirstQ', 'SecondQ', 'ThirdQ', 'FourthQ'], label = 'Zip Income Quartile')
hosp_region = gr.inputs.Radio(choices=['Midwest', 'Northeast', 'South', 'West'], label = 'Hospital Region')
hosp_division = gr.inputs.Radio(choices=['New England', 'Middle Atlantic', 'East North Central', 'West North Central', 'South Atlantic', 'East South Central', 'West South Central', 'Mountain', 'Pacific'], label = 'Hospital Division')
hosp_locteach = gr.inputs.Radio(choices=['Urban teaching', 'Urban nonteaching', 'Rural'], label= 'Hospital Location/Teaching')
hosp_bedsize = gr.inputs.Radio(choices=['Small', 'Medium', 'Large'], label= 'Hospital Bedsize')
h_contrl = gr.inputs.Radio(choices= ['Government_nonfederal', 'Private_invest_own', 'Private_not_profit'], label = 'Hospital Control')
pay = gr.inputs.Dropdown(choices= ['Private insurance', 'Medicare', 'Medicaid',  'Self-pay', 'No charge', 'Other'], label = 'Payee')
anemia = gr.inputs.Radio(choices=["No", "Yes"], label = 'Anemia')
atrial_fibrillation = gr.inputs.Radio(choices=["No", "Yes"], label = 'Atrial Fibrillation')
cancer = gr.inputs.Radio(choices=["No", "Yes"], label = 'Cancer')
cardiac_arrhythmias = gr.inputs.Radio(choices=["No", "Yes"], label = 'Cardiac Arrhythmias')
carotid_artery_disease = gr.inputs.Radio(choices=["No", "Yes"], label = 'Carotid Artery Disease') 
chronic_kidney_disease = gr.inputs.Radio(choices=["No", "Yes"], label = 'Chronic Kidney Disease')
chronic_pulmonary_disease = gr.inputs.Radio(choices=["No", "Yes"], label = 'Chronic Pulmonary Disease') 
coagulopathy =  gr.inputs.Radio(choices=["No", "Yes"], label = 'Coagulopathy')
depression = gr.inputs.Radio(choices=["No", "Yes"], label = 'Depression')
diabetes_mellitus = gr.inputs.Radio(choices=["No", "Yes"], label = 'Diabetes Mellitus')
drug_abuse = gr.inputs.Radio(choices=["No", "Yes"], label = 'Drug Abuse')
dyslipidemia = gr.inputs.Radio(choices=["No", "Yes"], label = 'Dyslipidemia')
endocarditis = gr.inputs.Radio(choices=["No", "Yes"], label = 'Endocarditis')
family_history = gr.inputs.Radio(choices=["No", "Yes"], label = 'Family History')
fluid_and_electrolyte_disorder = gr.inputs.Radio(choices=["No", "Yes"], label = 'Fluid and Electrolyte Disorder')
heart_failure = gr.inputs.Radio(choices=["No", "Yes"], label = 'Heart Failure')
hypertension = gr.inputs.Radio(choices=["No", "Yes"], label = 'Hypertension')
known_cad = gr.inputs.Radio(choices=["No", "Yes"], label = 'Known CAD')
liver_disease = gr.inputs.Radio(choices=["No", "Yes"], label = 'Liver Disease')
obesity = gr.inputs.Radio(choices=["No", "Yes"], label = 'Obesity')
peripheral_vascular_disease = gr.inputs.Radio(choices=["No", "Yes"], label = 'Peripheral Vascular Disease')
prior_cabg = gr.inputs.Radio(choices=["No", "Yes"], label = 'Prior CABG')
prior_icd = gr.inputs.Radio(choices=["No", "Yes"], label = 'Prior ICD')
prior_mi = gr.inputs.Radio(choices=["No", "Yes"], label = 'Prior MI')
prior_pci = gr.inputs.Radio(choices=["No", "Yes"], label = 'Prior PCI') 
prior_ppm = gr.inputs.Radio(choices=["No", "Yes"], label = 'Prior PPM')
prior_tia_stroke = gr.inputs.Radio(choices=["No", "Yes"], label = 'Prior TIA Stroke')
pulmonary_circulation_disorder = gr.inputs.Radio(choices=["No", "Yes"], label = 'Pulmonary Circulation Disorder') 
smoker = gr.inputs.Radio(choices=["No", "Yes"], label = 'Smoker')
valvular_disease = gr.inputs.Radio(choices=["No", "Yes"], label = 'Valvular Disease') 
weight_loss = gr.inputs.Radio(choices=["No", "Yes"], label = 'Weight Loss')
endovascular_tavr = gr.inputs.Radio(choices=["No", "Yes"], label = 'Endovascular TAVR')
transapical_tavr = gr.inputs.Radio(choices=["No", "Yes"], label = 'Transapical TAVR', default= 'Yes')


# Defining and launching the interface
gr.Interface(predict, [age, female, race, elective, aweekend, zipinc_qrtl, hosp_region, hosp_division, hosp_locteach,
            hosp_bedsize, h_contrl, pay, anemia, atrial_fibrillation, 
            cancer, cardiac_arrhythmias, carotid_artery_disease, 
            chronic_kidney_disease, chronic_pulmonary_disease, coagulopathy,
            depression, diabetes_mellitus, drug_abuse, dyslipidemia, endocarditis,
            family_history, fluid_and_electrolyte_disorder, heart_failure,
            hypertension, known_cad, liver_disease, obesity, peripheral_vascular_disease,
            prior_cabg, prior_icd, prior_mi, prior_pci, prior_ppm, prior_tia_stroke,
            pulmonary_circulation_disorder, smoker, valvular_disease, weight_loss,
            endovascular_tavr, transapical_tavr], 
            'label',
            live=True,
            title = "Predicting In-Hospital Mortality After TAVR Using Preoperative Variables and Penalized Logistic Regression",
            description = "The app below utilizes the finalized logistic regression model with an l2 penalty based on the manuscript by Alhwiti et al. The manuscript will be submitted to JACC: Cardiovascular Interventions. The data used for model building is all TAVR procedures between 2012 and 2019 as reported in the HCUP NIS database. <br><br> The purpose of the app is to provide evidence-based clinical support for interventional cardiology. <br> <br> For instruction on how to use the app and the encoding required for the variables,  please see <b>XYZ: insert website link here</b>.",
            css = 'https://bootswatch.com/5/journal/bootstrap.css').launch(share = True);
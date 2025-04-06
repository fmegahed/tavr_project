# pip install pycaret
from pandas.api.types import CategoricalDtype
import pandas as pd
import jinja2

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

# url = 'https://raw.githubusercontent.com/fmegahed/tavr_paper/main/data/example_data2.csv'
# download = requests.get(url).content

ex_data =pd.read_csv('example_data2.csv')
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
  ordinal_cat = CategoricalDtype(categories = ['FirstQ', 'SecondQ', 'ThirdQ', 'FourthQ'], ordered = True)
  df.zipinc_qrtl = df.zipinc_qrtl.astype(ordinal_cat)

  with urllib.request.urlopen('https://github.com/fmegahed/tavr_paper/blob/main/data/final_model.pkl?raw=true') as response, open('final_model.pkl', 'wb') as out_file:
    shutil.copyfileobj(response, out_file)

  model = load_model('final_model')

  pred = predict_model(model, df, raw_score=True)
  
  return {'Death %': round(100*pred['Score_Yes'][0], 2),
       'Survival %': round(100*pred['Score_No'][0], 2),
       'Predicting Death Outcome:': pred['Label'][0]}

# Defining the containers for each input
age = gr.Slider(minimum=18, maximum=100, value=60, label="Age")
female = gr.Dropdown(choices=["Female", "Male"],label = 'Sex')
race = gr.Dropdown(choices=['Asian or Pacific Islander', 'Black', 'Hispanic', 'Native American', 'White',  'Other'], label = 'Race')
elective = gr.Radio(choices=['Elective', 'NonElective'], label = 'Elective')
aweekend = gr.Radio(choices=["No", "Yes"], label = 'Weekend')
zipinc_qrtl = gr.Radio(choices=['FirstQ', 'SecondQ', 'ThirdQ', 'FourthQ'], label = 'Zip Income Quartile')
hosp_region = gr.Radio(choices=['Midwest', 'Northeast', 'South', 'West'], label = 'Hospital Region')
hosp_division = gr.Radio(choices=['New England', 'Middle Atlantic', 'East North Central', 'West North Central', 'South Atlantic', 'East South Central', 'West South Central', 'Mountain', 'Pacific'], label = 'Hospital Division')
hosp_locteach = gr.Radio(choices=['Urban teaching', 'Urban nonteaching', 'Rural'], label= 'Hospital Location/Teaching')
hosp_bedsize = gr.Radio(choices=['Small', 'Medium', 'Large'], label= 'Hospital Bedsize')
h_contrl = gr.Radio(choices= ['Government_nonfederal', 'Private_invest_own', 'Private_not_profit'], label = 'Hospital Control')
pay = gr.Dropdown(choices= ['Private insurance', 'Medicare', 'Medicaid',  'Self-pay', 'No charge', 'Other'], label = 'Payee')
anemia = gr.Radio(choices=["No", "Yes"], label = 'Anemia')
atrial_fibrillation = gr.Radio(choices=["No", "Yes"], label = 'Atrial Fibrillation')
cancer = gr.Radio(choices=["No", "Yes"], label = 'Cancer')
cardiac_arrhythmias = gr.Radio(choices=["No", "Yes"], label = 'Cardiac Arrhythmias')
carotid_artery_disease = gr.Radio(choices=["No", "Yes"], label = 'Carotid Artery Disease') 
chronic_kidney_disease = gr.Radio(choices=["No", "Yes"], label = 'Chronic Kidney Disease')
chronic_pulmonary_disease = gr.Radio(choices=["No", "Yes"], label = 'Chronic Pulmonary Disease') 
coagulopathy =  gr.Radio(choices=["No", "Yes"], label = 'Coagulopathy')
depression = gr.Radio(choices=["No", "Yes"], label = 'Depression')
diabetes_mellitus = gr.Radio(choices=["No", "Yes"], label = 'Diabetes Mellitus')
drug_abuse = gr.Radio(choices=["No", "Yes"], label = 'Drug Abuse')
dyslipidemia = gr.Radio(choices=["No", "Yes"], label = 'Dyslipidemia')
endocarditis = gr.Radio(choices=["No", "Yes"], label = 'Endocarditis')
family_history = gr.Radio(choices=["No", "Yes"], label = 'Family History')
fluid_and_electrolyte_disorder = gr.Radio(choices=["No", "Yes"], label = 'Fluid and Electrolyte Disorder')
heart_failure = gr.Radio(choices=["No", "Yes"], label = 'Heart Failure')
hypertension = gr.Radio(choices=["No", "Yes"], label = 'Hypertension')
known_cad = gr.Radio(choices=["No", "Yes"], label = 'Known CAD')
liver_disease = gr.Radio(choices=["No", "Yes"], label = 'Liver Disease')
obesity = gr.Radio(choices=["No", "Yes"], label = 'Obesity')
peripheral_vascular_disease = gr.Radio(choices=["No", "Yes"], label = 'Peripheral Vascular Disease')
prior_cabg = gr.Radio(choices=["No", "Yes"], label = 'Prior CABG')
prior_icd = gr.Radio(choices=["No", "Yes"], label = 'Prior ICD')
prior_mi = gr.Radio(choices=["No", "Yes"], label = 'Prior MI')
prior_pci = gr.Radio(choices=["No", "Yes"], label = 'Prior PCI') 
prior_ppm = gr.Radio(choices=["No", "Yes"], label = 'Prior PPM')
prior_tia_stroke = gr.Radio(choices=["No", "Yes"], label = 'Prior TIA Stroke')
pulmonary_circulation_disorder = gr.Radio(choices=["No", "Yes"], label = 'Pulmonary Circulation Disorder') 
smoker = gr.Radio(choices=["No", "Yes"], label = 'Smoker')
valvular_disease = gr.Radio(choices=["No", "Yes"], label = 'Valvular Disease') 
weight_loss = gr.Radio(choices=["No", "Yes"], label = 'Weight Loss')
endovascular_tavr = gr.Radio(choices=["No", "Yes"], label = 'Endovascular TAVR')
transapical_tavr = gr.Radio(choices=["No", "Yes"], label = 'Transapical TAVR', value= 'Yes')


# Defining and launching the interface
iface = gr.Interface(
    fn = predict, 
    inputs = [age, female, race, elective, aweekend, zipinc_qrtl, hosp_region, hosp_division, hosp_locteach,
            hosp_bedsize, h_contrl, pay, anemia, atrial_fibrillation, 
            cancer, cardiac_arrhythmias, carotid_artery_disease, 
            chronic_kidney_disease, chronic_pulmonary_disease, coagulopathy,
            depression, diabetes_mellitus, drug_abuse, dyslipidemia, endocarditis,
            family_history, fluid_and_electrolyte_disorder, heart_failure,
            hypertension, known_cad, liver_disease, obesity, peripheral_vascular_disease,
            prior_cabg, prior_icd, prior_mi, prior_pci, prior_ppm, prior_tia_stroke,
            pulmonary_circulation_disorder, smoker, valvular_disease, weight_loss,
            endovascular_tavr, transapical_tavr], 
    outputs = 'text',
    live=True,
    title = "Predicting In-Hospital Mortality After TAVR Using Preoperative Variables and Penalized Logistic Regression",
    description = """
        <p style="font-size:16px; line-height:1.6;">
        This app predicts in-hospital mortality after TAVR using a finalized logistic regression model with L2 penalty, based on national inpatient data from 2012–2019 (HCUP NIS).<br>
        <br>
        Published paper: 
        <a href="https://www.nature.com/articles/s41598-023-37358-9.pdf" target="_blank">
        Alhwiti, T., Aldrugh, S., & Megahed, F. M. (2023), <i>Scientific Reports</i>, 13(1), 10252.
        </a>
        </p>
    """,
    css = 'https://bootswatch.com/5/journal/bootstrap.css')

iface.launch()
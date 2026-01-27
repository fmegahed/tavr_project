import os
import shutil
import urllib.request
from pathlib import Path

import gradio as gr
import pandas as pd
from pandas.api.types import CategoricalDtype
from pycaret.classification import load_model, predict_model

# Optional: load example data (not required for predictions, but kept since it exists in your repo)
# If the file is missing, the app still runs.
try:
    _ex_data = pd.read_csv("example_data2.csv")
except Exception:
    _ex_data = None


MODEL_BASENAME = "final_model"          # pycaret load_model expects the basename
MODEL_FILE = f"{MODEL_BASENAME}.pkl"    # this should exist locally in your Space repo
MODEL_URL = "https://github.com/fmegahed/tavr_paper/blob/main/data/final_model.pkl?raw=true"

_MODEL = None


def _ensure_model_file() -> None:
    """
    Ensure final_model.pkl exists locally.
    If it is missing, try to download it once as a fallback.
    """
    if Path(MODEL_FILE).exists():
        return

    # Fallback: download if the repo file is missing for some reason
    with urllib.request.urlopen(MODEL_URL) as response, open(MODEL_FILE, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def _get_model():
    """
    Load and cache the PyCaret model once per process.
    """
    global _MODEL
    if _MODEL is None:
        _ensure_model_file()
        _MODEL = load_model(MODEL_BASENAME)
    return _MODEL


def predict(
    age,
    female,
    race,
    elective,
    aweekend,
    zipinc_qrtl,
    hosp_region,
    hosp_division,
    hosp_locteach,
    hosp_bedsize,
    h_contrl,
    pay,
    anemia,
    atrial_fibrillation,
    cancer,
    cardiac_arrhythmias,
    carotid_artery_disease,
    chronic_kidney_disease,
    chronic_pulmonary_disease,
    coagulopathy,
    depression,
    diabetes_mellitus,
    drug_abuse,
    dyslipidemia,
    endocarditis,
    family_history,
    fluid_and_electrolyte_disorder,
    heart_failure,
    hypertension,
    known_cad,
    liver_disease,
    obesity,
    peripheral_vascular_disease,
    prior_cabg,
    prior_icd,
    prior_mi,
    prior_pci,
    prior_ppm,
    prior_tia_stroke,
    pulmonary_circulation_disorder,
    smoker,
    valvular_disease,
    weight_loss,
    endovascular_tavr,
    transapical_tavr,
):
    df = pd.DataFrame.from_dict(
        {
            "age": [age],
            "female": [female],
            "race": [race],
            "elective": [elective],
            "aweekend": [aweekend],
            "zipinc_qrtl": [zipinc_qrtl],
            "hosp_region": [hosp_region],
            "hosp_division": [hosp_division],
            "hosp_locteach": [hosp_locteach],
            "hosp_bedsize": [hosp_bedsize],
            "h_contrl": [h_contrl],
            "pay": [pay],
            "anemia": [anemia],
            "atrial_fibrillation": [atrial_fibrillation],
            "cancer": [cancer],
            "cardiac_arrhythmias": [cardiac_arrhythmias],
            "carotid_artery_disease": [carotid_artery_disease],
            "chronic_kidney_disease": [chronic_kidney_disease],
            "chronic_pulmonary_disease": [chronic_pulmonary_disease],
            "coagulopathy": [coagulopathy],
            "depression": [depression],
            "diabetes_mellitus": [diabetes_mellitus],
            "drug_abuse": [drug_abuse],
            "dyslipidemia": [dyslipidemia],
            "endocarditis": [endocarditis],
            "family_history": [family_history],
            "fluid_and_electrolyte_disorder": [fluid_and_electrolyte_disorder],
            "heart_failure": [heart_failure],
            "hypertension": [hypertension],
            "known_cad": [known_cad],
            "liver_disease": [liver_disease],
            "obesity": [obesity],
            "peripheral_vascular_disease": [peripheral_vascular_disease],
            "prior_cabg": [prior_cabg],
            "prior_icd": [prior_icd],
            "prior_mi": [prior_mi],
            "prior_pci": [prior_pci],
            "prior_ppm": [prior_ppm],
            "prior_tia_stroke": [prior_tia_stroke],
            "pulmonary_circulation_disorder": [pulmonary_circulation_disorder],
            "smoker": [smoker],
            "valvular_disease": [valvular_disease],
            "weight_loss": [weight_loss],
            "endovascular_tavr": [endovascular_tavr],
            "transapical_tavr": [transapical_tavr],
        }
    )

    # Convert object columns to categorical
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype("category")

    # Convert ordinal column to ordered categorical
    ordinal_cat = CategoricalDtype(
        categories=["FirstQ", "SecondQ", "ThirdQ", "FourthQ"],
        ordered=True,
    )
    df["zipinc_qrtl"] = df["zipinc_qrtl"].astype(ordinal_cat)

    model = _get_model()
    pred = predict_model(model, df, raw_score=True)

    # These column names depend on how the model was trained and saved
    # This matches your original code: Score_Yes for death, Score_No for survival.
    return {
        "Death %": round(100 * float(pred["Score_Yes"].iloc[0]), 2),
        "Survival %": round(100 * float(pred["Score_No"].iloc[0]), 2),
        "Predicting Death Outcome": str(pred["Label"].iloc[0]),
    }


inputs = [
    gr.Slider(minimum=18, maximum=100, value=80, label="Age"),
    gr.Dropdown(choices=["Female", "Male"], value="Female", label="Sex"),
    gr.Dropdown(
        choices=[
            "Asian or Pacific Islander",
            "Black",
            "Hispanic",
            "Native American",
            "White",
            "Other",
        ],
        value="White",
        label="Race",
    ),
    gr.Radio(choices=["Elective", "NonElective"], value="Elective", label="Elective"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Weekend"),
    gr.Radio(
        choices=["FirstQ", "SecondQ", "ThirdQ", "FourthQ"],
        value="SecondQ",
        label="Zip Income Quartile",
    ),
    gr.Radio(
        choices=["Midwest", "Northeast", "South", "West"],
        value="South",
        label="Hospital Region",
    ),
    gr.Radio(
        choices=[
            "New England",
            "Middle Atlantic",
            "East North Central",
            "West North Central",
            "South Atlantic",
            "East South Central",
            "West South Central",
            "Mountain",
            "Pacific",
        ],
        value="South Atlantic",
        label="Hospital Division",
    ),
    gr.Radio(
        choices=["Urban teaching", "Urban nonteaching", "Rural"],
        value="Urban teaching",
        label="Hospital Location/Teaching",
    ),
    gr.Radio(choices=["Small", "Medium", "Large"], value="Large", label="Hospital Bedsize"),
    gr.Radio(
        choices=["Government_nonfederal", "Private_invest_own", "Private_not_profit"],
        value="Private_not_profit",
        label="Hospital Control",
    ),
    gr.Dropdown(
        choices=["Private insurance", "Medicare", "Medicaid", "Self-pay", "No charge", "Other"],
        value="Medicare",
        label="Payee",
    ),
    # Comorbidities
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Anemia"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Atrial Fibrillation"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Cancer"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Cardiac Arrhythmias"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Carotid Artery Disease"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Chronic Kidney Disease"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Chronic Pulmonary Disease"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Coagulopathy"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Depression"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Diabetes Mellitus"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Drug Abuse"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Dyslipidemia"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Endocarditis"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Family History"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Fluid and Electrolyte Disorder"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Heart Failure"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Hypertension"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Known CAD"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Liver Disease"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Obesity"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Peripheral Vascular Disease"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Prior CABG"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Prior ICD"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Prior MI"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Prior PCI"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Prior PPM"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Prior TIA Stroke"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Pulmonary Circulation Disorder"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Smoker"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Valvular Disease"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Weight Loss"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Endovascular TAVR"),
    gr.Radio(choices=["No", "Yes"], value="Yes", label="Transapical TAVR"),
]

description_html = """
<p style="font-size:16px; line-height:1.6;">
This app predicts in-hospital mortality after TAVR using a finalized logistic regression model with L2 penalty,
based on national inpatient data from 2012–2019 (HCUP NIS).<br>
<br>
Published paper:
<a href="https://www.nature.com/articles/s41598-023-37358-9.pdf" target="_blank">
Alhwiti, T., Aldrugh, S., & Megahed, F. M. (2023), <i>Scientific Reports</i>
</a>
</p>
"""

iface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    live=True,
    title="Predicting In-Hospital Mortality After TAVR Using Preoperative Variables and Penalized Logistic Regression",
    description=description_html,
    css="https://bootswatch.com/5/journal/bootstrap.css",
)

if __name__ == "__main__":
    # Works locally, in Docker, and on HF Spaces (PORT is commonly set by platforms)
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    iface.launch(server_name="0.0.0.0", server_port=port)

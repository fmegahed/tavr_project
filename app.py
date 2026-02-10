import os
import shutil
import urllib.request
from pathlib import Path

import gradio as gr
import pandas as pd
from pandas.api.types import CategoricalDtype
from pycaret.classification import load_model, predict_model

# Optional: load example data (not required for predictions, but kept since it exists in your repo)
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

    # Return dict with 0-1 scale for gr.Label confidence bars
    return {
        "Death": float(pred["Score_Yes"].iloc[0]),
        "Survival": float(pred["Score_No"].iloc[0]),
    }


# ---------------------------------------------------------------------------
# UI — gr.Blocks with tabs
# ---------------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft(), title="TAVR Mortality Prediction") as demo:

    gr.Markdown(
        """
        # Predicting In-Hospital Mortality After TAVR

        This app predicts in-hospital mortality after Transcatheter Aortic Valve
        Replacement (TAVR) using a logistic regression model (L2 penalty) trained
        on national inpatient data from 2012-2019 (HCUP NIS).

        **Paper:**
        [Alhwiti, T., Aldrugh, S., & Megahed, F. M. (2023), *Scientific Reports*](https://www.nature.com/articles/s41598-023-37358-9.pdf)
        """
    )

    with gr.Row():
        # ---- Left: inputs ----
        with gr.Column(scale=3):
            with gr.Tab("Patient Demographics"):
                with gr.Row():
                    age = gr.Slider(minimum=18, maximum=100, value=80, label="Age")
                    female = gr.Dropdown(choices=["Female", "Male"], value="Female", label="Sex")
                with gr.Row():
                    race = gr.Dropdown(
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
                    )
                    pay = gr.Dropdown(
                        choices=["Private insurance", "Medicare", "Medicaid", "Self-pay", "No charge", "Other"],
                        value="Medicare",
                        label="Payee",
                    )
                with gr.Row():
                    elective = gr.Radio(choices=["Elective", "NonElective"], value="Elective", label="Elective")
                    aweekend = gr.Radio(choices=["No", "Yes"], value="No", label="Weekend Admission")
                with gr.Row():
                    zipinc_qrtl = gr.Radio(
                        choices=["FirstQ", "SecondQ", "ThirdQ", "FourthQ"],
                        value="SecondQ",
                        label="Zip Income Quartile",
                    )

            with gr.Tab("Hospital Information"):
                with gr.Row():
                    hosp_region = gr.Radio(
                        choices=["Midwest", "Northeast", "South", "West"],
                        value="South",
                        label="Hospital Region",
                    )
                    hosp_bedsize = gr.Radio(
                        choices=["Small", "Medium", "Large"],
                        value="Large",
                        label="Hospital Bedsize",
                    )
                with gr.Row():
                    hosp_division = gr.Radio(
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
                    )
                with gr.Row():
                    hosp_locteach = gr.Radio(
                        choices=["Urban teaching", "Urban nonteaching", "Rural"],
                        value="Urban teaching",
                        label="Hospital Location/Teaching",
                    )
                    h_contrl = gr.Radio(
                        choices=["Government_nonfederal", "Private_invest_own", "Private_not_profit"],
                        value="Private_not_profit",
                        label="Hospital Control",
                    )

            with gr.Tab("Comorbidities"):
                with gr.Row():
                    anemia = gr.Radio(choices=["No", "Yes"], value="Yes", label="Anemia")
                    atrial_fibrillation = gr.Radio(choices=["No", "Yes"], value="Yes", label="Atrial Fibrillation")
                    cancer = gr.Radio(choices=["No", "Yes"], value="No", label="Cancer")
                with gr.Row():
                    cardiac_arrhythmias = gr.Radio(choices=["No", "Yes"], value="Yes", label="Cardiac Arrhythmias")
                    carotid_artery_disease = gr.Radio(choices=["No", "Yes"], value="No", label="Carotid Artery Disease")
                    chronic_kidney_disease = gr.Radio(choices=["No", "Yes"], value="Yes", label="Chronic Kidney Disease")
                with gr.Row():
                    chronic_pulmonary_disease = gr.Radio(choices=["No", "Yes"], value="Yes", label="Chronic Pulmonary Disease")
                    coagulopathy = gr.Radio(choices=["No", "Yes"], value="No", label="Coagulopathy")
                    depression = gr.Radio(choices=["No", "Yes"], value="No", label="Depression")
                with gr.Row():
                    diabetes_mellitus = gr.Radio(choices=["No", "Yes"], value="Yes", label="Diabetes Mellitus")
                    drug_abuse = gr.Radio(choices=["No", "Yes"], value="No", label="Drug Abuse")
                    dyslipidemia = gr.Radio(choices=["No", "Yes"], value="Yes", label="Dyslipidemia")
                with gr.Row():
                    endocarditis = gr.Radio(choices=["No", "Yes"], value="No", label="Endocarditis")
                    family_history = gr.Radio(choices=["No", "Yes"], value="No", label="Family History")
                    fluid_and_electrolyte_disorder = gr.Radio(choices=["No", "Yes"], value="Yes", label="Fluid & Electrolyte Disorder")
                with gr.Row():
                    heart_failure = gr.Radio(choices=["No", "Yes"], value="Yes", label="Heart Failure")
                    hypertension = gr.Radio(choices=["No", "Yes"], value="Yes", label="Hypertension")
                    known_cad = gr.Radio(choices=["No", "Yes"], value="Yes", label="Known CAD")
                with gr.Row():
                    liver_disease = gr.Radio(choices=["No", "Yes"], value="No", label="Liver Disease")
                    obesity = gr.Radio(choices=["No", "Yes"], value="Yes", label="Obesity")
                    peripheral_vascular_disease = gr.Radio(choices=["No", "Yes"], value="Yes", label="Peripheral Vascular Disease")
                with gr.Row():
                    prior_cabg = gr.Radio(choices=["No", "Yes"], value="Yes", label="Prior CABG")
                    prior_icd = gr.Radio(choices=["No", "Yes"], value="Yes", label="Prior ICD")
                    prior_mi = gr.Radio(choices=["No", "Yes"], value="Yes", label="Prior MI")
                with gr.Row():
                    prior_pci = gr.Radio(choices=["No", "Yes"], value="Yes", label="Prior PCI")
                    prior_ppm = gr.Radio(choices=["No", "Yes"], value="Yes", label="Prior PPM")
                    prior_tia_stroke = gr.Radio(choices=["No", "Yes"], value="Yes", label="Prior TIA/Stroke")
                with gr.Row():
                    pulmonary_circulation_disorder = gr.Radio(choices=["No", "Yes"], value="No", label="Pulmonary Circulation Disorder")
                    smoker = gr.Radio(choices=["No", "Yes"], value="No", label="Smoker")
                    valvular_disease = gr.Radio(choices=["No", "Yes"], value="Yes", label="Valvular Disease")
                with gr.Row():
                    weight_loss = gr.Radio(choices=["No", "Yes"], value="No", label="Weight Loss")

            with gr.Tab("Procedure"):
                with gr.Row():
                    endovascular_tavr = gr.Radio(choices=["No", "Yes"], value="Yes", label="Endovascular TAVR")
                    transapical_tavr = gr.Radio(choices=["No", "Yes"], value="Yes", label="Transapical TAVR")

        # ---- Right: output ----
        with gr.Column(scale=1):
            predict_btn = gr.Button("Predict", variant="primary")
            output = gr.Label(label="Prediction", num_top_classes=2)

    # Wire up the button
    all_inputs = [
        age, female, race, elective, aweekend, zipinc_qrtl,
        hosp_region, hosp_division, hosp_locteach, hosp_bedsize, h_contrl, pay,
        anemia, atrial_fibrillation, cancer, cardiac_arrhythmias,
        carotid_artery_disease, chronic_kidney_disease, chronic_pulmonary_disease,
        coagulopathy, depression, diabetes_mellitus, drug_abuse, dyslipidemia,
        endocarditis, family_history, fluid_and_electrolyte_disorder,
        heart_failure, hypertension, known_cad, liver_disease, obesity,
        peripheral_vascular_disease, prior_cabg, prior_icd, prior_mi,
        prior_pci, prior_ppm, prior_tia_stroke, pulmonary_circulation_disorder,
        smoker, valvular_disease, weight_loss,
        endovascular_tavr, transapical_tavr,
    ]

    predict_btn.click(fn=predict, inputs=all_inputs, outputs=output)


if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.launch(server_name="0.0.0.0", server_port=port)

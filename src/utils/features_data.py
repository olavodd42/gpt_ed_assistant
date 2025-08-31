from dataclasses import dataclass

@dataclass
class Feature:
    ed_ehr: list = (
        [
            "age", "gender",
                    
            "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", 
            "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d",
            
            "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", 
            "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", 
            "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", 
            "cci_Cancer2", "cci_HIV",  

            "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2", 
            "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", 
            "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
            "eci_Anemia", "eci_Alcohol", "eci_Drugs","eci_Psychoses", "eci_Depression",
            
            # VS
            "triage_temperature",
            "triage_heartrate",
            "triage_resprate", 
            "triage_o2sat",
            "triage_sbp",
            "triage_dbp",
            #RNHX
            "triage_pain",
            "triage_acuity",
            "chiefcomplaint",
        ]
    )
    ed_ehr_dummy: list = []
    ed_lab: list = (
        [
            #CBC
            'hematocrit',
            'white blood cells',
            'hemoglobin',
            'red blood cells',
            'mean corpuscular volume',
            'mean corpuscular hemoglobin',
            'mean corpuscular hemoglobin concentration',
            'red blood cell distribution width',
            'platelet count',
            'basophils',
            'eosinophils',
            'lymphocytes',
            'monocytes',
            'neutrophils',
            'red cell distribution width (standard deviation)',
            'absolute lymphocyte count',
            'absolute basophil count',
            'absolute eosinophil count',
            'absolute monocyte count',
            'absolute neutrophil count',
            'bands',
            'atypical lymphocytes',
            'nucleated red cells',
            #CHEM
            'urea nitrogen',
            'creatinine',
            'sodium',
            'chloride',
            'bicarbonate',
            'glucose (chemistry)',
            'potassium',
            'anion gap',
            'calcium, total',
            #COAG
            'prothrombin time', 'inr(pt)', 'ptt',
            #UA
            'ph (urine)',
            'specific gravity',
            'red blood count (urine)',
            'white blood count (urine)',
            'epithelial cells',
            'protein',
            'hyaline casts',
            'ketone',
            'urobilinogen',
            'glucose (urine)',
            #LACTATE
            'lactate',
            #LFTs
            'alkaline phosphatase',
            'asparate aminotransferase (ast)',
            'alanine aminotransferase (alt)',
            'bilirubin, total',
            'albumin',
            #LIPASE
            'lipase',
            #LYTES
            'magnesium', 'phosphate',
            #CARDIO,
            'ntprobnp', 'troponin t',
            #BLOOD_GAS
            'potassium, whole blood',
            'ph (blood gas)',
            'calculated total co2',
            'base excess',
            'po2',
            'pco2',
            'glucose (blood gas)',
            'sodium, whole blood',
            #TOX
            'ethanol',
            #INFLAMMATION
            'creatine kinase (ck)', 'c-reactive protein'
        ]
    )
    ed_lab_dummy: list = []

    ed_lab_idx: list = ([
        list(range(0,23)),#CBC
        list(range(23,32)),#CHEM
        list(range(32,35)),#COAG
        list(range(35,45)),#UA
        list(range(45,46)),#LACTATE
        list(range(46,51)),#LFTs
        list(range(51,52)),#LIPASE
        list(range(52,54)),#LYTES
        list(range(54,56)),#CARDIO
        list(range(56,64)),#BLOOD_GAS
        list(range(64,65)),#TOX
        list(range(65,67))#INFLAMMATION
    ])
    cost: dict = ({
        "CBC":30,
        "CHEM":60,    
        "COAG":48,
        "UA":40,
        "LACTATE":4,
        "LFTs":104,
        "LIPASE":100,
        "LYTES":89,
        "CARDIO":122,
        "BLOOG_GAS":12,
        "TOX":70,
        "INFLAMMATION":178
    })
    def __post_init__(self):
        self.ed_ehr_dummy = ['feature' + str(i+1) for i in range(len(self.ed_ehr))]
        self.ed_lab_dummy = ['feature' + str(i+len(self.ed_lab)+1) for i in range(len(self.ed_lab))]
    
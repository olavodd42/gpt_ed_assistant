# ==================== CONFIG BÁSICA ====================
TEXT_COL = "chiefcomplaint"
HASH_N = 20000
SVD_D  = 256
RANDOM_STATE = 42

# Faixas plausíveis (triagem e vitais)
VITAL_BOUNDS = {
    "triage_temperature": (30, 45),
    "triage_heartrate": (20, 220),
    "triage_resprate": (4, 60),
    "triage_o2sat": (50, 100),
    "triage_sbp": (50, 260),
    "triage_dbp": (20, 150),
    "triage_pain": (0, 10),

    "ed_temperature_last": (30, 45),
    "ed_heartrate_last": (20, 220),
    "ed_resprate_last": (4, 60),
    "ed_o2sat_last": (50, 100),
    "ed_sbp_last": (50, 260),
    "ed_dbp_last": (20, 150),
    "ed_pain_last": (0, 10),
}

# Colunas candidatas a log1p (cauda pesada)
LOG1P_CANDS = [
    # contagens de utilização prévia
    "n_ed_30d","n_ed_90d","n_ed_365d",
    "n_hosp_30d","n_hosp_90d","n_hosp_365d",
    "n_icu_30d","n_icu_90d","n_icu_365d",
    "n_med","n_medrecon",
    # labs comuns (se existirem nas suas tabelas)
    "white blood cells","platelet count","absolute neutrophil count",
    "c-reactive protein","creatine kinase (ck)","ntprobnp","lactate","glucose (chemistry)"
]
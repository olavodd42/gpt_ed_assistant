import os, sys
import pandas as pd
from src.utils.seed import seed
from src.data_prep.linearization import build_text_example, save_to_txt

seed()
from pathlib import Path
import pandas as pd
import json

BASE = Path(__file__).resolve().parent.parent

def load_standards():
    return pd.read_csv(BASE / "data" / "standards.csv").fillna("")

def load_demo_tender():
    return (BASE / "sample_data" / "sample_tender.txt").read_text(encoding="utf-8")

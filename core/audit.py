from pathlib import Path
import json
from datetime import datetime

PATH=Path(__file__).resolve().parent.parent/"data"/"audit_log.json"

def load_history():
    if not PATH.exists(): return []
    try: return json.loads(PATH.read_text(encoding="utf-8"))
    except: return []

def record_action(action, details):
    rows=load_history()
    rows.append({"timestamp":datetime.now().isoformat(timespec="seconds"),"action":action,**details})
    PATH.write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding="utf-8")

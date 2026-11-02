"""Atomic local artifacts: a reader sees either the old or the complete new file."""
import json
from pathlib import Path


def write_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(text,encoding='utf8');temporary.replace(path)


def write_json(path,value):
    write_text(path,json.dumps(value,ensure_ascii=False,indent=2))

"""Run validation-only selection and CanaryBench after Llama replacement."""

import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source_log = (
    ROOT
    / "results/eos_fixed/non_dp/llama3_2_1b_it/tmux_postprocess.log"
).read_text()
encoded = max(
    re.findall(r'b64decode\(\\?"([A-Za-z0-9+/=]+)\\?"', source_log),
    key=len,
)
program = base64.b64decode(encoded).decode()
program = program.replace(':02d"', ':02d}"')
program = program.replace("/proc/1204228", "/proc/1279119")
program = program.replace(
    "results/eos_fixed/non_dp/llama3_2_1b_it",
    "results/eos_fixed/non_dp/llama3_2_1b_it_eos_supervised_replacement",
)
program = program.replace(
    "steps=[3891,5188,6485,7782,9079,10376]",
    "steps=[1297*i for i in range(1,9)]",
)
program = program.replace("zip(range(3,9),steps)", "zip(range(1,9),steps)")
program = program.replace("list(range(3,9))", "list(range(1,9))")
program = program.replace('"unavailable_epochs":[1,2]', '"unavailable_epochs":[]')
program = program.replace(
    "from canarybench.prompt_format import build_generation_prompt",
    '''def build_generation_prompt(tok,text):
 return tok.apply_chat_template([{"role":"user","content":text}],tokenize=False,add_generation_prompt=True,date_string="31 Aug 2026")''',
)
program = program.replace(r'r"\\s+"', r'r"\s+"')
program = program.replace(
    r'x.strip().split("\\n")[0].strip()',
    'x.strip().splitlines()[0].strip().strip(chr(96)).strip().'
    'strip(chr(34)+chr(39)).rstrip(".:;!,").strip()',
)
audit_patch = '''import canarybench.prompt_format as _pf, metrics.base as _mb
def _gp(tok,text):
 return tok.apply_chat_template([{"role":"user","content":text}],tokenize=False,add_generation_prompt=True,date_string="31 Aug 2026")
def _st(tok,rec):
 if rec.get("task")=="lm" or not rec.get("input","").strip(): return "",rec.get("target","")
 p=_gp(tok,rec["input"])
 f=tok.apply_chat_template([{"role":"user","content":rec["input"]},{"role":"assistant","content":rec.get("target","")}],tokenize=False,add_generation_prompt=False,date_string="31 Aug 2026")
 return p,f
_pf.build_generation_prompt=_gp;_pf.build_supervised_text=_st
_mb.build_generation_prompt=_gp;_mb.build_supervised_text=_st
'''
program = program.replace(
    "from types import SimpleNamespace",
    audit_patch + "from types import SimpleNamespace",
)
exec(program)

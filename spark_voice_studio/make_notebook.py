#!/usr/bin/env python3
"""make_notebook.py - build Spark_Voice_Studio.ipynb from the step scripts.

Each step becomes a Markdown cell (the script's docstring) and a code cell that runs the script with
arguments you can edit in place. The code is not copied into the notebook; the notebook calls the same
files the menu calls, so a fix in a .py file is a fix in both. A final section shows how to call a step's
functions directly when you want to step through one in the notebook.

    python make_notebook.py            # writes Spark_Voice_Studio.ipynb beside the scripts
"""
import ast, re
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
STEPS = sorted(ROOT.glob("step*_*.py"))
DEFAULT_ARGS = {
    "step1": "",
    "step2": "",
    "step3": "--mode clone",
    "step4": "--i-am-the-speaker",
    "step5": "--i-am-the-speaker --epochs 3",
    "step6": "sample.txt",
    "step7": "--i-am-the-speaker --keywords MBSE SysML",
}


def docstring(path):
    tree = ast.parse(path.read_text())
    return ast.get_docstring(tree) or path.name


nb = nbf.v4.new_notebook()
cells = [nbf.v4.new_markdown_cell(
    "# Spark Voice Studio\n\nConnect to the DGX Spark, download and load a voice model, record your voice, fine-tune, "
    "play back text, and tune on keywords. Each section runs one of the `stepN_*.py` scripts; edit the `ARGS` line "
    "and run the cell. The recorder (step 4) and the player (step 6) need a terminal-style prompt, which Jupyter "
    "provides through `input()`.\n\nEdit `studio.yaml` first: Spark host, user, key, and your name as owner."),
    nbf.v4.new_code_cell("import subprocess, sys, json\nfrom pathlib import Path\nROOT = Path.cwd()\n\n"
                         "def run(script, args=''):\n    \"\"\"Run a step script with live output in the notebook.\"\"\"\n"
                         "    cmd = [sys.executable, str(ROOT / script), *args.split()]\n"
                         "    print('$', ' '.join(cmd[1:]))\n"
                         "    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)\n"
                         "    for line in p.stdout: print(line, end='')\n"
                         "    return p.wait()\n\n"
                         "def status():\n    p = ROOT / 'state.json'\n    print(json.dumps(json.loads(p.read_text()), indent=1) if p.exists() else 'no state yet')\n\nstatus()")]
for s in STEPS:
    key = s.name.split("_")[0]
    doc = docstring(s)
    title = doc.splitlines()[0]
    body = "\n".join(doc.splitlines()[1:]).strip()
    cells.append(nbf.v4.new_markdown_cell(f"## {title}\n\n{body}"))
    cells.append(nbf.v4.new_code_cell(f"ARGS = '{DEFAULT_ARGS.get(key, '')}'\nrun('{s.name}', ARGS)"))
cells.append(nbf.v4.new_markdown_cell(
    "## Working inside a step\n\nThe scripts are importable. To speak one sentence through the loaded model, or to check the "
    "connection without the whole step, call the pieces directly."))
cells.append(nbf.v4.new_code_cell(
    "from studio.common import Spark, cfg, state\nimport requests\n\nsp = Spark()\nprint(sp.run('nvidia-smi --query-gpu=name,memory.used --format=csv,noheader', venv=False, stream=False))\n"
    "url = sp.url()\nprint(requests.get(url + '/health', timeout=5).json())\n"
    "r = requests.post(url + '/speak', json={'text': 'This is a test of the studio from the notebook.'}, timeout=300)\n"
    "open('out/notebook_test.wav', 'wb').write(r.content); print('out/notebook_test.wav', r.headers.get('X-RTF'), 'x real time')\n"
    "from IPython.display import Audio; Audio('out/notebook_test.wav')"))
cells.append(nbf.v4.new_markdown_cell("## Assessment report\n\nMean scores per model from step 6, so a fine-tuned checkpoint can be compared with the zero-shot clone on the same text."))
cells.append(nbf.v4.new_code_cell("run('step6_playback_text.py', '--report')"))
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
out = ROOT / "Spark_Voice_Studio.ipynb"
nbf.write(nb, out)
print(f"wrote {out} with {len(cells)} cells")

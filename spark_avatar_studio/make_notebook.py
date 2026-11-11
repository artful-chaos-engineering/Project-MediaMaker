#!/usr/bin/env python3
"""make_notebook.py - build Spark_Avatar_Studio.ipynb from the step scripts.

Each step becomes a Markdown cell (the script's docstring) and a code cell that runs the script with
arguments you can edit in place. The code is not copied into the notebook; the notebook calls the same
files the menu calls, so a fix in a .py file is a fix in both. A final section shows how to call a step's
functions directly when you want to step through one in the notebook.

    python make_notebook.py            # writes Spark_Avatar_Studio.ipynb beside the scripts
"""
import ast, re
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
STEPS = sorted(ROOT.glob("step*_*.py"))
DEFAULT_ARGS = {
    "step1": "",
    "step2": "",
    "step3": "",
    "step4": "--i-am-the-subject --video",
    "step5": "--i-am-the-subject",
    "step6": "sample.txt",
    "step7": "--i-am-the-subject --audio data/audio/sample.wav",
    "step8": "lesson --mode color --color #0B1F3A",
    "step9": "--report",
}


def docstring(path):
    tree = ast.parse(path.read_text())
    return ast.get_docstring(tree) or path.name


nb = nbf.v4.new_notebook()
cells = [nbf.v4.new_markdown_cell(
    "# Spark Avatar Studio\n\nConnect to the DGX Spark, download the lip-sync and background models, capture photos or an idle video of "
    "yourself, build the avatar, turn text into speech with your cloned voice, render the talking avatar, replace the background, and "
    "tune the result. Each section runs one of the `stepN_*.py` scripts; edit the `ARGS` line and run the cell. Steps 4 and 9 prompt "
    "through `input()`, which Jupyter provides. Step 4 needs a camera on the machine running the notebook.\n\n"
    "Edit `studio.yaml` first: Spark host, user, key, your name as owner, and the background you want."),
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
cells.append(nbf.v4.new_markdown_cell("## Look at the result\n\nPlay a finished render, or the background preview from step 5, without leaving the notebook."))
cells.append(nbf.v4.new_code_cell(
    "from IPython.display import Video, Image, display\nname = 'lesson'   # the render name\n"
    "p = ROOT / 'out' / name / f'{name}_final.mp4'\nprint(p, 'exists' if p.exists() else 'not rendered yet')\n"
    "display(Video(str(p), embed=False, width=640)) if p.exists() else None\n"
    "from studio.common import cfg\nbp = ROOT / 'avatars' / cfg()['avatar']['name'] / 'bg_preview.png'\n"
    "display(Image(str(bp))) if bp.exists() else print('no background preview yet (step 5)')"))
cells.append(nbf.v4.new_markdown_cell("## Assessment report\n\nScores per render with the settings that produced them, so a sweep from step 9 shows which settings won."))
cells.append(nbf.v4.new_code_cell("run('step9_assess_tune.py', '--report')"))
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
out = ROOT / "Spark_Avatar_Studio.ipynb"
nbf.write(nb, out)
print(f"wrote {out} with {len(cells)} cells")

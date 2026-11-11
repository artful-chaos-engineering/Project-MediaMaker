# Spark Voice Studio

Seven Python tools, a text menu, and a notebook that take a DGX Spark from a bare SSH login to a fine-tuned
clone of your own voice that reads text files back to you for assessment, then tunes on the words it gets wrong.

## How it is built

The studio runs in two places. The menu, the microphone capture (step 4) and the playback with scoring (step 6)
run where you sit, on the Mac. Everything that needs the GPU runs on the Spark: the model server, the downloads,
training. Step 1 syncs this folder to the Spark and builds a Python venv there; after that every step talks to
the Spark through `studio/common.py`, which runs commands over SSH (paramiko) or in-process when
`studio.yaml` says the host is `local` (that is, when you run the studio on the Spark itself).

"Load the model into memory" is a resident process, `studio/model_server.py`, which holds a Qwen3-TTS model in
the Spark's unified memory and speaks on request over HTTP (`/health`, `/load`, `/speak`). It has two modes:
`clone`, the Base model with a voice prompt from your reference clip, which works the moment you have recorded
anything, and `custom`, a fine-tuned checkpoint that answers to your speaker name. Step 6 scores whichever is
loaded, so you can compare the two on the same text.

The voice model is Qwen3-TTS (Apache 2.0). It is the one open model in late 2026 with both strong zero-shot cloning
and a documented single-speaker fine-tuning path: `finetuning/prepare_data.py` encodes your recordings with the
12 Hz tokenizer, `finetuning/sft_12hz.py` trains, and each epoch writes a `checkpoint-epoch-N` folder that loads
as a complete model. Step 5 drives those two scripts; step 7 resumes from the last checkpoint with keyword
recordings added.

| Step | Script | Runs on | What it does |
|---|---|---|---|
| 1 | `step1_connect_spark.py` | Mac → Spark | SSH check, GPU check (sm_120 kernels, the NVRTC filterbank test), venv with cu130 PyTorch and the packages, code sync |
| 2 | `step2_download_model.py` | Spark | Qwen3-TTS Base, its tokenizer, Whisper, and the Qwen3-TTS repository (fine-tuning scripts) |
| 3 | `step3_load_model.py` | Spark | starts the model server and loads a model in `clone` or `custom` mode |
| 4 | `step4_capture_audio.py` | Mac | guided reading of 40 sentences (or import of existing recordings), silence trim, reference clip selection, upload |
| 5 | `step5_train_voice.py` | Spark | Whisper transcripts for untranscribed clips, `train.jsonl`, tokenizer codes, fine-tuning run, checkpoint recorded |
| 6 | `step6_playback_text.py` | Mac | each paragraph of a text file spoken, played, scored 1 to 5 three ways; wrong words collected; report by model |
| 7 | `step7_keyword_tune.py` | Mac + Spark | keyword sentences recorded, training resumed with them; or a lexicon fix (written form → spoken form) applied before speaking |

`studio.py` is the menu (`python studio.py`; or `python studio.py 1 2 3` to run steps unattended).
`make_notebook.py` writes `Spark_Voice_Studio.ipynb`, one section per step, each cell calling the same script with
an editable `ARGS` line, plus a cell that speaks one sentence inline and a cell with the assessment report.

## Setup

On the Mac: `pip install -r requirements.txt` (pyyaml, paramiko, sounddevice, soundfile, numpy, requests, nbformat).
Edit `studio.yaml`: Spark host, user, SSH key, your name as `owner`, and the model size (1.7B default; 0.6B is faster).
Then `python studio.py` and choose 1. The Spark side needs nothing beforehand but SSH and the stock DGX OS.

## Order of a first session

1 connect · 2 download (about 8 GB) · 4 capture (40 sentences, 15 minutes) · 3 load in `clone` mode · 6 play `sample.txt`
and score it (this is your zero-shot baseline) · 5 train (3 epochs; an hour or two on the GB10 for 40 clips) ·
3 load in `custom` mode · 6 play the same text and compare the report · 7 keyword tune on the words both got wrong.

## Rules built in

The studio clones the owner's voice only. Steps 4, 5 and 7 refuse to run without `--i-am-the-speaker`, and
`data/voice/manifest.json` records the owner. Published audio from a cloned voice should be disclosed as synthetic.

## Known edges

The Qwen3-TTS fine-tuning script arguments follow the project's guide (`init_model_path`, `output_model_path`,
`train_jsonl`, `batch_size`, `lr`, `num_epochs`, `speaker_name`); if the project renames one, `--help` on the Spark
shows the current names and `training.extra_args` in `studio.yaml` passes anything extra through. On the Spark,
PyTorch wheels can hit the NVRTC sm_121 error inside the first spectrogram; step 1 tests for it and points to the
fix in *How to Produce a Voice Clone on a DGX Spark*, Table 1. Nothing here has run on a GB10 yet; the pieces that
could be tested without one (menu, import, trimming, reference selection, lexicon, local runner, notebook build) were.

# Spark Avatar Studio

Nine Python tools, a text menu, and a notebook that take a DGX Spark from a bare SSH login to a talking avatar of you:
your photos or idle video in, text in, a lip-synced video out, with the background removed or replaced.
It is the sibling of Spark Voice Studio: the voice studio makes the speech, this one makes the face that speaks it.

## Is it possible?

Yes. The pipeline is: text -> your cloned voice (Spark Voice Studio) -> audio -> **LatentSync 1.6** rewrites the mouth region of
your own footage so the lips match the audio -> **BiRefNet** (or a chroma key) cuts you out -> ffmpeg lays you over the new background.
All of it runs on the Spark, offline once the models are downloaded.

## Models on the Spark

| Job | Model | Licence | Size | Notes |
|---|---|---|---|---|
| Lip-sync | LatentSync 1.6 (ByteDance), latent-diffusion UNet at 512x512 | Apache 2.0 | about 5 GB checkpoint, about 18 GB of memory in use | the quality model; takes real footage, keeps your head movement, rewrites the mouth |
| Audio features for lip-sync | Whisper-tiny (inside the LatentSync checkpoints) | MIT | 75 MB | |
| Image codec for lip-sync | stabilityai/sd-vae-ft-mse | MIT | 0.3 GB | fetched once by step 2 |
| Background removal | BiRefNet (ZhengPeng7/BiRefNet; the -portrait variant is person-tuned) | MIT | about 0.9 GB | AI matte, works on any room |
| Background removal, alternative | chroma key (no model) | n/a | n/a | green or blue screen footage; the cleanest edges |
| Voice | Qwen3-TTS from Spark Voice Studio | Apache 2.0 | about 8 GB | optional here; you can bring your own WAV |
| Optional, not wired in | Wav2Lip (fast, soft mouth), EchoMimicV3 / Wan2.2-S2V (photo to moving head and hands, 34 to 61 GB) | various | | see *How to Produce a Digital Avatar on the DGX Spark* |

128 GB of unified memory holds the voice server, LatentSync and BiRefNet together with room to spare; LatentSync loads
per render (its inference script builds the pipeline each run), so the first seconds of every render are model loading.

## What you need to supply

**Best: a 30 to 60 second idle video.** You looking at the lens, relaxed, blinking naturally, mouth mostly closed, small
head movements. LatentSync keeps whatever the head does and rewrites the mouth, so real footage looks alive. Step 5 finds the
best loop window in it so long speech does not show a seam.

**Enough to start: photos.** One sharp, front-facing photo is enough for a first avatar. The head will be still with a gentle camera
drift, which reads as a talking portrait, not a person. Step 4 asks for five shots (front neutral, slight smile, mouth open as in "ahh",
head turned 10 degrees left and right); only the front neutral one drives the still-photo avatar today, the others are spares and a
record of the likeness.

Technical requirements for either, all checked by step 4 before upload:
- one face, facing the camera within about 15 degrees, eyes open and on the lens
- face at least 400 px tall (1080p with your face filling about 40% of the frame height; 4K is better, LatentSync works at 512x512 on the face crop)
- sharp on the eyes, evenly lit, no hard side shadow across the mouth, no glare on glasses
- hands and microphones away from the chin; hair off the mouth; beard fine, but not changed between takes
- 24 or 25 fps for video; for a replaced background, film against a plain wall or a green screen and keep a little space around your shoulders

## Can the background be removed or replaced?

Yes, in steps 7 and 8, and previewed in step 5. Modes: **keep**, **transparent** (ProRes 4444 .mov for editors, or VP9 .webm),
**color**, **gradient**, **image** (a slide, office or brand backdrop), **video** (a looping clip) and **blur**. Two engines:
`birefnet` cuts you out of any room; `chroma` keys a green or blue screen and is cleaner at the hair line if you filmed on one. The
background stage runs after the lip-sync, on the finished video, so you can change the background in minutes without redoing the
slow part (step 8 reads `<name>_raw.mp4`). Edge controls: `choke` pulls the edge in against halos of the old room, `alpha_smooth`
steadies flicker, `feather` softens the edge. Expect the hair edge to be the weak point of an AI matte on a busy room.

## How it is built

The menu and the camera run on the Mac; everything that needs the GPU runs on the Spark. Step 1 syncs this folder to the Spark and builds a venv
with the cu130 PyTorch wheels. Each GPU step runs as `python stepN... --here` on the Spark through `studio/common.py`, which uses SSH (paramiko), or runs
in-process when `studio.yaml` says the host is `local`.

| Step | Script | Runs on | What it does |
|---|---|---|---|
| 1 | `step1_connect_spark.py` | Mac -> Spark | SSH, GPU check, venv, LatentSync cloned with torch pins removed, packages |
| 2 | `step2_download_models.py` | Spark | LatentSync checkpoint and Whisper-tiny, the VAE, BiRefNet |
| 3 | `step3_load_models.py` | Spark | loads BiRefNet, times a 1080p matte, checks the files; `--smoke` renders a 3 s tone onto your avatar |
| 4 | `step4_capture_media.py` | Mac | guided photos and idle video from the webcam, or import; quality report; consent record; upload |
| 5 | `step5_build_avatar.py` | Spark | normalise footage, find the loop window (or build a drifting still from a photo), background preview |
| 6 | `step6_make_audio.py` | Mac | text file -> your voice via the Spark Voice Studio server, joined with pauses; writes WAV and SRT; or import a WAV |
| 7 | `step7_render_avatar.py` | Mac + Spark | cut audio at silences into chunks, ping-pong footage, LatentSync per chunk, join, original audio back, background stage |
| 8 | `step8_background.py` | Spark | new background on a finished render without re-running the lip-sync |
| 9 | `step9_assess_tune.py` | Mac | automatic checks, watch-and-score, sweeps of lip offset and quality settings, apply the winner |

`studio.py` is the menu (`python studio.py`). `make_notebook.py` writes `Spark_Avatar_Studio.ipynb`, one section per step with an editable `ARGS` line that
calls the same script, plus cells to play a render and show the background preview.

## Order of a first session

1 connect · 2 download · 4 capture (video if you can, photos if not) · 5 build, open `bg_preview.png` · 3 `--smoke` · 6 text to speech · 7 render a 30 s piece ·
9 `--auto`, then `--sweep offset` if the lips are early or late · 8 try other backgrounds on the same render.

## Rules built in

The studio animates the owner's face only. Steps 4, 5 and 7 refuse to run without `--i-am-the-subject`; step 4 writes `consent.json`; step 7 stamps every file's metadata with
"Synthetic video: lip-synced footage of the owner driven by a cloned voice". Disclose synthetic video when you publish it.

## Known edges

- Nothing here has run on a GB10 yet. Tested here without a GPU: the background stage (every mode, chroma engine), loop finding, chunking, restart from the chunk cache, footage
  cursor, the audio step against a stand-in server, the menu and the notebook build. The LatentSync call uses the project's documented command line (`scripts.inference` with
  `--unet_config_path`, `--inference_ckpt_path`, `--inference_steps`, `--guidance_scale`, `--video_path`, `--audio_path`, `--video_out_path`); run `--help` on the Spark if a flag moved.
- BiRefNet is run through Hugging Face `transformers` with `trust_remote_code=True`; the first run downloads its code.
- A photo-only avatar looks like a portrait. Expect the best results from real footage.
- Positive `audio_offset` moves the lips later relative to the sound; step 9 `--sweep offset` finds the value by eye.

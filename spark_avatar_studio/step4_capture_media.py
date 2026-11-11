#!/usr/bin/env python3
"""Step 4 - Capture photos and an idle video of yourself, check them, upload them.

Runs where the camera is (your Mac). The avatar is built from what you give it here:
  * best: a 30 to 60 s idle video (--video) - you looking at the lens, relaxed, blinking, small natural movement,
    mouth mostly closed. LatentSync keeps your head movement and rewrites only the mouth, so real footage looks alive.
  * enough to start: photos only (--photos). One good front-facing photo works; the avatar will have a still head with
    gentle camera drift. Five shots are requested so you can pick the best and keep spares.
Every file is checked (one face, face large enough, sharp, evenly lit, centred) and reported before upload.

    python step4_capture_media.py --i-am-the-subject --photos          # guided stills from the webcam
    python step4_capture_media.py --i-am-the-subject --video           # idle take, studio.yaml capture.video_seconds
    python step4_capture_media.py --i-am-the-subject --import me.jpg idle.mov
    python step4_capture_media.py --check-only                         # re-run the checks on what is already saved
"""
import argparse, datetime, json, shutil, subprocess, sys, time
from pathlib import Path
import cv2, numpy as np
from studio.common import ROOT, Spark, cfg, say, require_owner, set_state

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}; PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def src_dir():
    d = ROOT / "data" / "avatar" / cfg()["avatar"]["name"] / "source"; d.mkdir(parents=True, exist_ok=True); return d


def check_frame(img):
    """Return (metrics dict, list of problems) for one BGR image."""
    h, w = img.shape[:2]; g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if casc.empty():                                   # opencv-python without its data files: judge the whole frame only
        m = {"width": w, "height": h, "faces": -1, "sharp": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1), "brightness": round(float(g.mean()), 1)}
        return m, ["face detector unavailable (pip install opencv-python); face checks skipped"]
    faces = casc.detectMultiScale(g, 1.1, 6, minSize=(max(60, h // 10),) * 2)
    m = {"width": w, "height": h, "faces": len(faces)}; bad = []
    if len(faces) != 1:
        bad.append("no face found (look at the lens, add light)" if len(faces) == 0 else f"{len(faces)} faces in frame; only you should be visible")
        return m, bad
    x, y, fw, fh = faces[0]; roi = g[y:y + fh, x:x + fw]
    m.update(face_height=int(fh), face_frac=round(fh / h, 3), sharp=round(float(cv2.Laplacian(roi, cv2.CV_64F).var()), 1),
             brightness=round(float(roi.mean()), 1), cx=round((x + fw / 2) / w, 3), cy=round((y + fh / 2) / h, 3))
    if fh < 300: bad.append(f"face only {fh} px tall; move closer or use a higher resolution (aim for 400+)")
    if m["face_frac"] < 0.22: bad.append("face is small in the frame; fill more of it")
    if m["sharp"] < 60: bad.append("face looks soft; focus on your eyes, steady the camera")
    if not 70 <= m["brightness"] <= 190: bad.append("face too dark" if m["brightness"] < 70 else "face too bright; reduce glare")
    if abs(m["cx"] - 0.5) > 0.2: bad.append("face is off-centre")
    return m, bad


def check_file(p):
    p = Path(p)
    if p.suffix.lower() in VIDEO_EXT:
        cap = cv2.VideoCapture(str(p)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); res = []
        for i in np.linspace(0, max(n - 1, 0), 8).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
            if ok: res.append(check_frame(f))
        cap.release()
        if not res: return {"file": p.name, "kind": "video", "problems": ["could not read the video"]}
        bads = sorted({b for _, bb in res for b in bb}); fr = float(np.mean([m["faces"] == 1 for m, _ in res]))
        return {"file": p.name, "kind": "video", "frames_sampled": len(res), "face_found_frac": fr, "problems": bads,
                "face_height": int(np.median([m.get("face_height", 0) for m, _ in res]))}
    img = cv2.imread(str(p))
    if img is None: return {"file": p.name, "kind": "photo", "problems": ["could not read the image (HEIC? convert to JPG first)"]}
    m, bad = check_frame(img); return {"file": p.name, "kind": "photo", **m, "problems": bad}


def grab(cam, tries=6):
    for _ in range(tries): ok, f = cam.read()          # first frames are dark while exposure settles
    return f


def open_cam(c):
    cam = cv2.VideoCapture(c["capture"]["camera"])
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080); cam.set(cv2.CAP_PROP_FPS, c["capture"]["fps"])
    if not cam.isOpened(): sys.exit("camera not available; use --import with files you already have")
    return cam


def photos(c):
    shots = [l.split("|", 1) for l in (ROOT / "prompts" / "shot_list.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    cam = open_cam(c); d = src_dir()
    for key, text in shots:
        key, text = key.strip(), text.strip()
        while True:
            print(f"\n  {key}: {text}"); input("  press Enter, then hold the pose through the countdown ")
            for n in (3, 2, 1): print(f"  {n}", flush=True); time.sleep(1)
            f = grab(cam); out = d / f"photo_{key}.png"; cv2.imwrite(str(out), f)
            r = check_file(out); print("  ", "OK" if not r["problems"] else "; ".join(r["problems"]))
            if not r["problems"] or input("  retake? (Enter = yes, n = keep it): ").strip().lower() == "n": break
    cam.release()


def video(c):
    cam = open_cam(c); fps = c["capture"]["fps"]; secs = c["capture"]["video_seconds"]; d = src_dir(); raw = d / "idle_raw.avi"
    print(f"\n  Idle take, {secs} s. Look at the lens, relax your face, blink normally, keep your mouth mostly closed, small head movements only.")
    input("  press Enter, then settle in during the countdown "); [print(f"  {n}", flush=True) or time.sleep(1) for n in (5, 4, 3, 2, 1)]
    f = grab(cam); h, w = f.shape[:2]
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)); t0 = time.time(); n = 0
    while time.time() - t0 < secs:
        ok, f = cam.read()
        if ok: vw.write(f); n += 1
    vw.release(); cam.release()
    real = n / (time.time() - t0); say(f"{n} frames, camera delivered {real:.1f} fps; resampling to {fps}")
    out = d / "idle.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-r", f"{real:.3f}", "-i", str(raw), "-r", str(fps), "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(out)], check=True)
    raw.unlink()


def main():
    c = cfg(); ap = argparse.ArgumentParser()
    ap.add_argument("--i-am-the-subject", action="store_true"); ap.add_argument("--photos", action="store_true"); ap.add_argument("--video", action="store_true")
    ap.add_argument("--import", dest="imp", nargs="*"); ap.add_argument("--check-only", action="store_true"); args = ap.parse_args()
    d = src_dir()
    if not args.check_only:
        require_owner(args)
        if args.imp:
            for f in args.imp: shutil.copy(f, d / Path(f).name)
        if args.photos: photos(c)
        if args.video: video(c)
        if not (args.imp or args.photos or args.video): sys.exit("choose --photos, --video or --import FILES")
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXT | PHOTO_EXT)
    if not files: sys.exit(f"nothing in {d}")
    report = [check_file(p) for p in files]
    for r in report: print(f"  {r['file']:32} {r['kind']:5}  {'OK' if not r['problems'] else '; '.join(r['problems'])}")
    (d.parent / "report.json").write_text(json.dumps(report, indent=1))
    if args.check_only: return
    (d.parent / "consent.json").write_text(json.dumps({"owner": c["owner"], "subject_is_owner": True, "date": datetime.date.today().isoformat(),
        "statement": "The person in these files is the owner named here, or has given the owner written consent to animate their likeness."}, indent=1))
    sp = Spark(); name = c["avatar"]["name"]; sp.sync_code()
    for p in files + [d.parent / "report.json", d.parent / "consent.json"]:
        sp.put(p, f"avatars/{name}/source/{p.name}" if p.parent == d else f"avatars/{name}/{p.name}")
    set_state(media_uploaded=True, avatar=name); say(f"uploaded {len(files)} files to the Spark; next: step 5 builds the avatar")


if __name__ == "__main__":
    main()

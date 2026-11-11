#!/usr/bin/env python3
"""Step 7 - Fine-tune on keywords.

The words the model gets wrong (flagged in step 6, or given here) become targeted training: the script
writes short sentences around each keyword, you record them, they go to data/keywords/, and training
resumes from the last checkpoint with the keyword set included. It also keeps data/keywords/lexicon.json,
a written-form -> spoken-form map, for the words that are better fixed in the text than in the model
(acronyms read as letters, product names respelled); step 6 applies it before speaking.

    python step7_keyword_tune.py --i-am-the-speaker                       # keywords from step 6's wordlist
    python step7_keyword_tune.py --i-am-the-speaker --keywords MBSE SysML Vriesenga --takes-per-keyword 3
    python step7_keyword_tune.py --lexicon MBSE "M B S E" SysML "sis mel"   # text-side fix, no recording
    python step7_keyword_tune.py --i-am-the-speaker --train-only           # recordings exist; just train
"""
import argparse, json, random, sys
from pathlib import Path
from studio.common import ROOT, Spark, cfg, say, set_state, state, require_owner

KW = ROOT / "data" / "keywords"; LEX = KW / "lexicon.json"; WORDS = KW / "wordlist.txt"
FRAMES = ["Let me say that again: {k}.", "The word is {k}, and it matters here.", "In this module we cover {k} in detail.",
          "Write {k} on the board and underline it.", "Most people get {k} wrong the first time.", "{k} is the term; remember it.",
          "We will return to {k} in the next section.", "Ask yourself what {k} means for your program."]


def main():
    c = cfg(); ap = argparse.ArgumentParser()
    ap.add_argument("--i-am-the-speaker", action="store_true"); ap.add_argument("--keywords", nargs="*")
    ap.add_argument("--takes-per-keyword", type=int, default=3); ap.add_argument("--lexicon", nargs="*", help="pairs: WRITTEN SPOKEN ...")
    ap.add_argument("--train-only", action="store_true"); ap.add_argument("--epochs", type=int, default=2); args = ap.parse_args()
    KW.mkdir(parents=True, exist_ok=True)
    if args.lexicon:
        lex = json.loads(LEX.read_text()) if LEX.exists() else {}
        pairs = args.lexicon
        if len(pairs) % 2: sys.exit("--lexicon takes pairs: WRITTEN SPOKEN")
        for w, s in zip(pairs[::2], pairs[1::2]): lex[w] = s
        LEX.write_text(json.dumps(lex, indent=1, ensure_ascii=False)); say(f"lexicon now has {len(lex)} entries"); Spark().put(LEX, "data/keywords/lexicon.json")
        if not args.i_am_the_speaker: return
    require_owner(args)
    if not args.train_only:
        kws = args.keywords or (WORDS.read_text().split() if WORDS.exists() else [])
        if not kws: sys.exit("no keywords: pass --keywords or flag words in step 6")
        sentences = [random.choice(FRAMES).format(k=k) for k in kws for _ in range(args.takes_per_keyword)]
        random.shuffle(sentences)
        sfile = KW / "keyword_sentences.txt"; sfile.write_text("\n".join(sentences) + "\n")
        say(f"{len(sentences)} sentences for {len(kws)} keywords -> recording session")
        # Reuse step 4's recorder, pointed at the keywords folder.
        import step4_capture_audio as cap
        cap.DATA = KW
        sys.argv = ["step4", "--i-am-the-speaker", "--sentences", str(sfile), "--takes", str(len(sentences))]
        cap.main()
    sp = Spark()
    for f in sorted(KW.glob("utt*")): sp.put(f, f"data/keywords/{f.name}")
    init = state().get("last_checkpoint")
    say(f"resuming training from {init or 'the Base model'} with the keyword set")
    import step5_train_voice as tr
    run = f"kw_{len(list(KW.glob('utt*.wav')))}"
    sys.argv = ["step5", "--i-am-the-speaker", "--include-keywords", "--run", run, "--epochs", str(args.epochs)] + (["--init", f"{sp.home}/{init}"] if init else [])
    tr.main()
    set_state(keyword_run=run)
    say("keyword tuning done; load the new checkpoint with step 3 --mode custom and re-run step 6 on the same text to compare")


if __name__ == "__main__":
    main()

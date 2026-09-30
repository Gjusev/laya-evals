# Brag Plan: laya-evals

## What is this app?
An open-source calibration auditor and cheap judge for eval sets: it scores eval items
with a CPU-scale System 1 decision model, then checks whether the model's confidence
meant anything before you automate on it.

## The angle
We set out to check the upstream benchmark claims and could not break them. Every
published number came back within 0.0003 — at most one item in 300 — on a laptop CPU,
under an independently written harness. The deadpan joke: the most suspicious thing
about a confidence score is how much it changes meaning between a 3-option and a
20-option question, and nobody audited that. We did, and the audit is the product.

## Hook (first 2-3 seconds)
Black screen. Mono type, one line: "we tried to prove their benchmark was wrong."
Beat. Second line: "we couldn't." Both hold long enough to read twice.

## Key moments (the middle)
- The four-claim table filling in one row at a time: published vs re-run, each row
  stamped `reproduces`, deltas of +0.0003 / 0.0000 / 0.0000 / −0.0003.
- The threshold split: "0.81 confidence" over a 3-option question vs the same score
  over 20 options. The 20-option gate reads 0.9944. Same model, same score, opposite odds.
- The CI terminal: `check_regression.py` lines resolving `OK ... delta +0.0000`,
  exit 0, build green — calibration held, accuracy held.

## Outro / punchline
"judge cheap. audit the confidence." set in the editorial serif over warm bone,
then the wordmark `laya-evals` and the GitHub slug. Final beat: "all four claims
reproduced. we're still suspicious of the confidence."

## User flow worth showing
none — landing-page only (the product is a library; the page shows the gate terminal,
the reproduction table, and the threshold meters, which are the product's face).

## Tone
- Preset: deadpan
- Creative direction: forensic — the comedy of receipts, delivered flat
- Interpretation: long holds, one idea per scene, monospace numbers treated as
  evidence, no exclamation anywhere, motion limited to one reveal per beat.

## Format: landscape — 1920x1080
## Duration: 21 seconds

## Visual identity (from the project)
- Background: #F7F6F3 (warm bone), cards #FFFFFF with 1px #EAEAEA borders
- Accent: pale blue #E1F3FE / #1F6C9F for data, pale green #EDF3EC / #346538 for "reproduces"
- Text: #111111 ink, #787774 muted
- Display font: Newsreader / Instrument Serif (editorial serif, tight tracking)
- Body font: Geist Sans / SF Pro Display stack
- Strongest visual element: the faux-terminal gate output and the reproduces table

## Share copy (draft)
We tried to break laya's published benchmark numbers and failed: all four claims
reproduced within 0.0003 on a laptop CPU. Then we audited the confidence and found
the real story — 0.81 is a safe gate at 3 options and barely enough at 20.

## Audio direction
- Role: sparse professional accents over a quiet warm bed
- Music: warm minimal bed, low, slightly dry
- Music treatment: fade in under the hook, hold low, small swell on the table
  completing, resolve under the outro
- Music cue guidance: no preset cue file; cues at scene boundaries (~0s, ~5s, ~11s, ~17s)
- Audio-reactive treatment: none
- SFX posture: sparse — a soft tick per table row, one quiet keystroke tick on the
  terminal scene, one resolved tone on exit 0
- Audio-coupled moments: table rows arriving on ticks; terminal lines typing; exit 0 tone
- Restraint rule: no whooshes, no risers, nothing perky; the audio is a lab, not a launch

## Storyboard

### Scene 1 — the accusation — 4s
Warm bone background, centered mono line: "we tried to prove their benchmark was wrong."
Hold 1.2s. Second line fades in below: "we couldn't." Hold. Nothing else on screen.
Sequential/interaction: yes — two lines, one beat apart
Audio intent: quiet, curious, slightly conspiratorial
Audio-coupled idea: typed-text ticks on each line
Music: warm bed fading in, barely there
Transition mood: soft → Scene 2

### Scene 2 — the receipts — 6s
The four-claim table (from the landing page, restaged full-frame): rows arrive one by
one, each stamped `reproduces` in pale green. Deltas in monospace: +0.0003, 0.0000,
0.0000, −0.0003. Caption appears after the last row: "one item in 300. laptop cpu."
Sequential/interaction: yes — four rows, one tick each, caption last
Audio intent: matter-of-fact, accumulating weight
Audio-coupled idea: soft tick per row
Music: bed holds low, tiny swell on caption
Transition mood: clean → Scene 3

### Scene 3 — the real story — 6s
Two panels side by side, same number, different fates. Left: "3 options" with 0.8074
and a nearly-full meter, tag "automate". Right: "20 options" with 0.9944, tag "barely
enough". Header line: "the same confidence score, two different sports." Bottom line:
"thresholds don't transfer. so we advise per shape."
Sequential/interaction: yes — panels appear left then right, header first
Audio intent: the turn — the one moment of tension
Audio-coupled idea: none, let the hold do it
Music: bed thins to almost silence
Transition mood: soft → Scene 4

### Scene 4 — the gate — 3s
The terminal window: three `OK` lines resolving (real gate output), then
"exit 0 — accuracy held, calibration held."
Sequential/interaction: yes — lines type in sequence
Audio intent: resolved, closed
Audio-coupled idea: keystroke ticks, one resolved tone on exit 0
Music: bed returns, warm
Transition mood: clean → Scene 5

### Scene 5 — outro — 2s
Serif line: "judge cheap. audit the confidence." Wordmark `laya-evals`, GitHub slug
`github.com/Gjusev/laya-evals`. Small final mono line: "all four claims reproduced.
still suspicious of the confidence."
Sequential/interaction: none — one composed frame
Audio intent: dry, final
Audio-coupled idea: none
Music: resolves and fades out

**Music mood for this video:** deadpan
**Audio summary:** a quiet warm bed under four forensic beats, sparse ticks marking
evidence, one resolved tone at the end — nothing perky, nothing loud.

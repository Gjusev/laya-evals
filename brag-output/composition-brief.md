# Hyperframes Composition Brief: laya-evals

## Objective
Create a short launch-style brag video for laya-evals.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 21 seconds

## Source Material
- Project root: `C:\Code Main\laya-evals`
- Primary files read: `docs/index.html` (landing page, the visual identity source), `README.md`, `results/reproduction.json`, `scripts/check_regression.py` output
- Product name: laya-evals
- Tagline / strongest claim: "Judge every eval item for cents. Then check whether the confidence meant anything."
- Key UI or visual moment to recreate: the faux-terminal gate output (`OK ... delta +0.0000`, exit 0) and the reproduction table with pale-green `reproduces` stamps
- Copy that must appear verbatim:
  - "we tried to prove their benchmark was wrong." / "we couldn't."
  - "MASSIVE intent, English 0.783 -> 0.7833", "MASSIVE intent, 13 languages 0.451 -> 0.4510", "XNLI, English 0.860 -> 0.8600", "XNLI, 14 languages 0.731 -> 0.7307" (each stamped `reproduces`)
  - "one item in 300. laptop cpu."
  - "3 options -> gate 0.8074" vs "20 options -> gate 0.9944" ("thresholds don't transfer. so we advise per shape.")
  - "exit 0 — accuracy held, calibration held"
  - "judge cheap. audit the confidence." + `github.com/Gjusev/laya-evals`
  - "all four claims reproduced. still suspicious of the confidence."

## Creative Direction
- Tone preset: deadpan
- Creative direction: forensic — the comedy of receipts, delivered flat
- Interpretation: long holds, one idea per scene, monospace numbers as evidence, no
  exclamation anywhere, motion limited to one reveal per beat, generous empty space
- Angle: we tried to break the upstream benchmark and couldn't (deltas within 0.0003 on
  CPU); the real product is the audit that found the confidence score changes meaning
  with option count
- Hook: "we tried to prove their benchmark was wrong." beat "we couldn't."
- Outro / punchline: "judge cheap. audit the confidence." + "all four claims reproduced.
  still suspicious of the confidence."
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals
  - Unrelated visual redesign
  - Perky motion or perky audio

## Visual Identity
- Background: #F7F6F3 (warm bone); cards #FFFFFF, 1px #EAEAEA borders, 12px radius
- Text: #111111 ink, #787774 muted
- Accent: pale green #EDF3EC / #346538 (reproduces), pale blue #E1F3FE / #1F6C9F (data)
- Display font: Georgia / Times New Roman italic (editorial serif stand-in; no network
  fonts at render time)
- Body font: system sans (Helvetica Neue / Segoe UI stack)
- Mono font: Consolas / Cascadia Mono (all numbers, all evidence)
- Visual references from the project: faux-OS window chrome (three light dots), the
  claims table, threshold meters, pale pill tags

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. the accusation — 4s — two centered mono lines on warm bone
2. the receipts — 6s — four-row table arriving row by row, reproduced stamps, caption
3. the real story — 6s — two panels: 3 options gate 0.8074 vs 20 options gate 0.9944
4. the gate — 3s — terminal window typing three OK lines, exit 0
5. outro — 2s — serif tagline, wordmark, slug, final mono line

## Audio
- Audio role: sparse professional accents over a quiet warm bed
- Audio arc: bed fades in under the hook, holds low, thins at scene 3, returns warm
  at the gate, resolves under the outro
- Music: `assets/music/happy-beats-business-moves-vol-1-by-ende-dot-app.mp3` (treated
  as a low bed, ~20% volume; deadpan posture comes from restraint, not track choice)
- Music treatment: low throughout; small swell when the table completes; fade out
  under the outro
- Music cue guidance: detect at composition time via `npx hyperframes beats`; optional
  hints at ~5s (table start), ~11s (panels), ~17s (gate resolve)
- Audio-reactive treatment: none (deadpan; motion does not chase the music)
- Audio-coupled moments:
  - scene 1 lines — typed-text ticks
  - scene 2 rows — one soft tick per row (rows are readable text: snap to beats only
    if beats are >= 0.6s apart, else every other beat)
  - scene 4 terminal lines — keystroke ticks; one resolved tone on exit 0
- SFX selection guidance: `assets/sfx/sfx-analysis.md`; prefer low high-frequency-risk
  files for repeated moments
- Exact SFX choice: composition decides filenames, timestamps, density, volume
- Audio files: music copied to `brag-output/composition/assets/music/`; SFX chosen by
  the composition from the brag skill's sfx library

## Hyperframes Instructions
Requirements:
- Show at least one real UI/copy element: the gate terminal and the reproduction table
  are restaged from `docs/index.html` with real output values
- All text readable in the final render; reading floors respected (label ~0.8s settled,
  sentence ~0.3s/word)
- 15-25 seconds total; plan is 21s
- One paused GSAP timeline registered at `window.__timelines["laya-evals-brag"]`
- Deterministic: no Math.random, no repeat:-1, no runtime network (system font stacks;
  GSAP via CDN as the minimal skeleton does)
- Run `npx hyperframes check` (the single gate) before render

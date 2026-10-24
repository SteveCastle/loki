# Repose lab: state and next steps

Read SEARCH_SPACE.md (what has been tried), LEADERBOARD.md (scores), FAVORITE.md (what to ship now).

## Running

```
cd tools/repose-lab
python setup.py && python reps.py 640     # only if data/ is missing: builds the 41 cases (CPU, RTMPose)
python lab.py run                         # runs every queued generation not yet finished; safe to kill and rerun
touch PAUSE                               # stop after the current chunk of 40 jobs; rm PAUSE and run again to resume
python lab.py board                       # rewrite LEADERBOARD.md
python evosheet.py ...                    # side-by-side sheets for eyeballing (see the file)
```

Needs the batch build of the engine (`tools/loki-retouch`, `--jobs`, built in target/release; the app's
~/bin/loki-retouch.exe is untouched) and the models in ~/bin/models. Images go to data/ (gitignored); scores to
results/. A generation is `generations/genN.json`: its strategies (names in prompts.py), steps, seeds, and why.

## Process (evolutionary)

Each generation: keep the 2-3 fittest as parents (plus `base` as a control), add mutations of each (one change),
crossovers (combine winning clauses / reps), and 1-2 new ideas from an untried dimension, so it does not settle in a
local maximum. Eyeball one sheet per survivor for quirks. Update SEARCH_SPACE.md (what and why), FAVORITE.md (best
app-feasible) and the next-steps list below.

## Where it stopped (2026-10-24)

Gen 7 finished: x1idref confirmed best (0.760 at 4 seeds); nothing to change in the app. Stopped at the user's request after gen 7 (4 steps, 4 seeds: base, v2lead, xd-lead, x1idref, xdl-idref, x1-idcrop,
xdl-idcrop). Gen 8 (per-image sizes) is written and parked as `generations/next-gen8.json.todo`: rename it to
`gen8.json` and `python lab.py run` to resume. x1idref (FAVORITE.md) is implemented in the app (uncommitted):
`media-server/tasks/scene_finish.go` grounding with {pose}/{self}, `scene_posewords.go`, identity ref in
`GenerateLayerMask`, editor mirror in `scene-api.ts` + preview in `panel.tsx`. White paper: claude.ai artifact
"Repose: Pose Transfer Without Losing the Person".

## Next steps

- [x] gen 2 finish (warpref 0.40 pose / warpbase 0.23: collages pull back to the original pose)
- [x] gen 3: v2lead best; wording variants within noise
- [x] gen 4: x1delta = best fidelity (delta words); crossovers flat
- [ ] gen 5 (queued): seed-noise check, steps 4/12, shift 0.5/1.0, ref-resolution 384/1024, ocapsule
- [ ] gen 6: cross x1delta with the best gen 5 settings; delta words + v2lead order; best-of-N 4 seeds on 2 finalists
- [ ] keep the window profile from turning toward the camera: try "keep the direction the body faces"
- [ ] finalists at app settings (full size, 25 steps) on the user's 2 real cases, then ship FAVORITE.md
- Ideas not yet tried:
  - [x] delta words (x1delta) - works
  - [ ] coloured capsule (ocapsule, queued in gen 5)
  - put the figure on a darkened copy of the person's own silhouette (not the whole picture: v2over failed)
  - the person cut out on black as an extra identity reference (<image3>)
  - a two-pass: repose at low strength, then a second pass "restore the face and body of <image2>"
  - shift (flow schedule) and steps 4/6/12 on the winner; seeds 4 on finalists
  - check the winner at full size / 25 steps (what the app runs) before shipping

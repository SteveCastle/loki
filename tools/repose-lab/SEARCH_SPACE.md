# Repose search space

What has been tried for the Scene editor's **Repose** finish (`media-server/tasks/scene_finish.go`
`sceneFinishGrounding[FinishRepose]`, mirrored in `src/renderer/components/scene/scene-api.ts`), and how it scored.
Engine: loki-retouch (Qwen Image 2.1, instruction edit; <image1> = picture, <image2>... = references). Read this before
adding a strategy so a run knows where the search has been. Live numbers: `LEADERBOARD.md`. Code: `prompts.py`.

## Goal

Move the character into the drawn pose **without changing who they are**: same face, same body type (weight, build,
musculature, fat, curves), exactly two arms and two legs joined naturally, no ghost of the old limbs, background kept.
The user is especially sensitive to body/face changes.

## How a strategy is scored

- Cases (`setup.py`): 3 people from real scenes (window: profile, curvy; pool: slim, frontal, arms out; suit: frontal,
  curvy, hand at face, legs cropped at the thigh) x 13 synthetic poses built by rotating their own detected limbs (bone
  lengths kept; out-of-frame ankles read short so shins are stretched to 0.9 thigh) + the user's 2 real mannequins = 41.
  Poses: armsup, wave, hips, tpose, knee, behindhead, reach, point, folded, step, headhand, combo1, combo2.
- pose: RTMPose on the output; share of the joints the target moves (elbows/wrists/knees/ankles > 0.3 torso from the
  source) that land within 0.25 torso of target, best of as-is / left-right swapped. The source itself scores 0.
- face / body: masked SSIM vs the source in the face box and the torso+hips box, arms (old and new place) masked out,
  best over small shifts. Drops when the face or body shape changes (also when the whole person moves or is redrawn).
- fitness = 0.5 pose + 0.25 face + 0.25 body. Eyeball sheets every round too: extra limbs, broken hands, painted
  figures and odd anatomy don't always show in the numbers.

## Dimensions

1. **Grounding wording** (the instruction around the user's words)
2. **Pose words**: none / hand-written per case / automatic from the mannequin's joints (`posewords.py`, ports to the app)
3. **Reference representation** (`reps.py`): black (OpenPose on black, what the app sends) / white / thick / stick
   (black line figure on white) / capsule (grey filled mannequin with body-width limbs) / pair (current pose | new pose) /
   warp (a cut-and-paste collage of the person's own limbs rotated into the pose) / none
4. **Structure**: edit the picture with the pose as reference, or edit the warp collage with the picture as reference
5. **Sampling**: steps (8 for screening; ranking matched 12), size (640 long edge for screening)

## History

### Gen 0: the shipped prompt and its context (by hand, 2 scenes)
- Shipped `repose` (= `base`): "Make the person in <image1> do the exact same pose as shown in <image2>. Match the
  positions of the head, arms, hands and legs. Keep the person exactly as they are in <image1>: the same face, hair,
  clothing, body and style. ..." Third arms (old limbs kept, new ones added: "keep exactly as they are" reads as keep the
  limbs) and shrunken limbs (the mannequin's shin was 40% of real, "match the positions" made the model shrink the leg).
- Before 8f76e64 the prompt said skeleton/keypoint/repose: the engine painted the skeleton on the person. Banned words
  (test `TestSceneReposeGroundingAvoidsDrawingWords`): skeleton, keypoint, openpose, repose, lines, colour.
- The Repose layer mask was only the stick figure grown 6%: the old body showed around the new one. Fixed in the editor
  (`finishShapes`: cover the detected character / the figure's padded box). Separate from the prompt work.

### Gen 1 (17 cases, 768 px, 12 steps; pose metric only)
| strategy | pose | notes |
|---|---|---|
| base | 0.56 | extra arms; 1/3 don't move |
| v2 | 0.64 | pose = directions not lengths; exactly two arms/legs; moved limb leaves its place; identity list |
| v2w | **0.73** | v2 + hand-written pose words |
| v2auto | 0.68 | v2 + automatic pose words |
| v3 | 0.52 | heavy "same weight, build, musculature... do not slim, enlarge, smooth or idealise": suppresses movement |
| v3auto | 0.63 | v3 + automatic words |
| v2over | 0.03 | the figure drawn over the darkened picture: the engine redraws the overlay, the person is lost (face 0.33) |

### Gen 2 (17 cases, 640 px, 8 steps; + fidelity)
| strategy | pose | face | body | no move /34 |
|---|---|---|---|---|
| base | 0.58 | 0.682 | 0.738 | 8 |
| v2w | 0.68 | 0.708 | 0.723 | 6 |
| v2auto | 0.66 | 0.703 | 0.738 | 6 |
| v2auto-white | 0.51 | 0.639 | 0.588 | 10 |
| v2auto-stick | 0.52 | 0.572 | 0.535 | 11 |
| v2auto-capsule | 0.64 | 0.668 | 0.611 | 7 |
| v2auto-pair | 0.16 | 0.755 | 0.787 | 24 (mostly copies the picture) |
| warpref | 0.40 | 0.741 | 0.723 | 12 (also a third arm on suit-tpose: hand left at the face) |
| warpbase | 0.23 | 0.818 | 0.831 | 22 (edits the collage back to the original pose: high "fidelity" = no change) |

Takeaways so far: OpenPose-on-black is the best figure; white backgrounds and plain line art hurt both pose and
fidelity; side-by-side "from/to" makes it not move; the person's own body as a collage (reference or input) pulls the
result back to the original pose. Note: high face/body with low pose usually means "didn't move", read them together.
Metric blind spots seen: targets whose hands leave the frame score 0 even when right (suit-tpose); third arms don't
lower face/body much. 8 steps blow out bright backgrounds (window): confirm finalists at app settings. Words help (+0.08 to +0.1 pose) at no fidelity cost. Strong
body-preservation language trades pose for little gain.

### Gen 3 (41 cases, 640 px, 8 steps, 2 seeds, batch engine; angle metric added after, all rescored)
Parent v2auto; children each change one thing.
| strategy | change | fitness | pose | angle | face | body | no move /82 |
|---|---|---|---|---|---|---|---|
| v2lead | pose words lead the prompt ("New pose for the person in <image1>: ...") | **0.750** | 0.76 | 0.81 | 0.718 | 0.720 | 8 |
| v2only | + "Only the arms and legs placed differently in <image2> move; the rest stays" | 0.745 | 0.74 | 0.79 | 0.724 | 0.732 | 8 |
| v4 | + v3's body-type list (weight, build, musculature, fat, curves) without "do not slim" | 0.740 | 0.73 | 0.78 | 0.725 | 0.724 | 9 |
| v2auto | parent | 0.737 | 0.74 | 0.77 | 0.712 | 0.723 | 9 |
| v2auto-thick | figure lines x2 | 0.694 | 0.71 | 0.78 | 0.659 | 0.630 | 10 |
| base | shipped | 0.684 | 0.62 | 0.71 | 0.679 | 0.731 | 16 |
| v2words-noref | words only, no figure | 0.513 | 0.44 | 0.56 | 0.530 | 0.523 | 22 |

The figure is essential (words alone collapse); thick lines cost fidelity. Wording variants cluster within ~0.013.
Visual: no third arms in the leaders; the window profile sometimes turns toward the camera on big arm moves, which
flattens her curves (body 0.72-0.76 vs 0.93 for base on window-combo2): a body-type change the user would notice.

### Gen 4 (crossovers of the gen 3 leaders)
| strategy | change | fitness* | pose | face | body |
|---|---|---|---|---|---|
| x1 | v2lead x v2only (lead words + only-moved-limbs) | 0.737 | 0.72 | 0.728 | 0.729 |
| x2 | x1 + body-type list | 0.737 | 0.72 | 0.729 | 0.728 |
| x1delta | words for the moved limbs only ("Change only this: ...") + only-moved-limbs | 0.743 | 0.71 | **0.753** | **0.761** |
| x1loose | "take the pose shown in" instead of "do the exact same pose as shown in" | see board | | | |
| x1idref | the picture again as <image3> "for their face and body" | see board | | | |
(*before the angle rescore.) Delta words: best fidelity so far at the same pose: naming only what changes leaves the
rest of the person alone. Crossing near-identical parents gains nothing: wording has plateaued.

### Selection (best-of-N), simulated on gen 3/4 data
Generate 2 seeds and keep the one with the best (pose + angle + face + body): +0.05 fitness (x1 0.737 -> 0.791;
v2lead 0.750 -> 0.788; base 0.684 -> 0.737). Selecting by pose match alone gains little (+0.02) and lowers body: the
pick must include fidelity. The app can compute all four (it has the picture, the mannequin and RTMPose on the server).

Gen 4 (final, rescored): x1idref 0.754 (pose 0.70, angle 0.77, face 0.779, body 0.769) = best; x1loose 0.744;
x1delta 0.743.

### Gen 5: sampling, representation, seed noise (x1 unless named)
| variant | fitness | pose | angle | face | body | reading |
|---|---|---|---|---|---|---|
| v2lead, seeds 3000+ | 0.745 | 0.70 | 0.76 | 0.808 | 0.706 | vs 0.750 on seeds 1000+: fitness noise ~0.01, single metrics swing ~0.05-0.09 |
| 4 steps | 0.740 | 0.72 | 0.78 | 0.731 | 0.729 | = 8 = 12 steps: use 4 for screening |
| 12 steps | 0.740 | 0.74 | 0.78 | 0.718 | 0.719 | |
| shift 1.0 / 0.5 | 0.734 / 0.729 | | | | | no gain (default 0.69) |
| ref-resolution 384 | 0.710 | 0.75 | 0.81 | 0.652 | 0.632 | more pose, much less fidelity |
| ref-resolution 1024 | 0.708 | 0.61 | 0.69 | 0.772 | 0.769 | more fidelity, less pose |
| ocapsule (figure over coloured body mass) | 0.722 | 0.71 | 0.78 | 0.706 | 0.686 | mass costs fidelity again |
--ref-resolution also sizes <image1>: the trade above is mostly the picture's resolution. Engine now takes per-image
sizes in --jobs ("sizes") to test the picture at 1024 with the figure small (gen 8).

### Gen 6: x1delta line
| strategy | change | fitness | pose | angle | face | body |
|---|---|---|---|---|---|---|
| xd-lead | delta words in the leading position | 0.750 | 0.74 | 0.79 | 0.736 | 0.739 |
| xd-face | + "head and body keep facing the same way, same angle to camera" | 0.727 | 0.66 | 0.75 | 0.743 | 0.756 |
| xd-face-body | + body-type list | 0.728 | 0.66 | 0.75 | 0.743 | 0.757 |
The facing clause costs pose (0.66) for no fidelity gain: dropped.

### Gen 7 (4 steps, 4 seeds; the confirmation run)
| strategy | fitness | pose | face | body | no move /164 |
|---|---|---|---|---|---|
| x1idref (shipped to the app) | **0.760** | 0.70 | 0.807 | 0.787 | 21 |
| xdl-idref (delta words + identity ref) | 0.758 | 0.69 | **0.812** | **0.794** | 21 |
| v2lead | 0.754 | **0.74** | 0.757 | 0.735 | 17 |
| xd-lead | 0.751 | 0.72 | 0.766 | 0.749 | 19 |
| x1-idcrop (tight crop of the person as identity ref) | 0.721 | 0.70 | 0.732 | 0.702 | 19 |
| xdl-idcrop | 0.720 | 0.68 | 0.735 | 0.715 | 20 |
| base | 0.699 | 0.63 | 0.716 | 0.746 | 29 |
The identity reference holds up at 4 seeds; the whole picture beats a tight crop of the person. Delta words add nothing
on top of it (and would need the original pose stored). Visual: no third arms in the leaders; base produced a detached
arm on window-wave. pool-user is enlarged by every strategy: the preset mannequin is bigger than her (mannequin scale,
not prompt). Gen 8 (per-image sizes) parked: generations/next-gen8.json.todo.

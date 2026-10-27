# Current favorite (update every generation)

**As of gen 6: `x1idref`.** Fitness 0.754 (shipped prompt: 0.684), and the best face/body fidelity of the high-pose
strategies (face 0.779, body 0.769). Pending the 4-seed confirmation in gen 7 (`results/gen7--*`).

Inputs to the engine:
- `<image1>` the picture (crop) to edit, as today
- `<image2>` the mannequin as the OpenPose figure on black, as today (`renderPaintAlone`)
- `<image3>` **new:** the same crop as `<image1>` again, as an identity reference

## Grounding

`{pose}` = the mannequin's pose in words (`posewords.py describe`), `{what}` = the user's words.

```
New pose for the person in <image1>: {pose}. Make the person in <image1> do the exact same pose as shown in <image2>:
{what}. <image2> only shows the pose: where the head, shoulders, elbows, hands, hips, knees and feet go. Follow the
direction each part of the body points in <image2>, but keep this person's own body: the same proportions, limb
lengths, build and size in the frame as in <image1>, so it is clearly the same body, only moved. Only the arms and legs
that are placed differently in <image2> move; the rest of the body stays exactly where it is. They have exactly two
arms, two hands and two legs, each joined naturally to the body; an arm or leg that moves leaves its old place, which
then shows the background or their clothing instead. <image3> is the same picture again, for their face and body. Keep
their identity exactly: the same face, hair, skin and figure, and the same clothing, which drapes and folds naturally
with the new pose. Keep the background, the lighting, the camera angle and the framing of <image1> unchanged.
```

Example `{pose}` for the user's window mannequin: "both arms folded across the chest; the leg on the left of the
picture lifted, the knee raised high toward the left of the picture and the foot off the ground; the leg on the right
of the picture standing straight". With no `{what}` the ": {what}" folds away; with no mannequin (a hand-drawn figure)
the leading "New pose ..." sentence goes.

## Also worth shipping: best-of-N

Generating 2 seeds and keeping the one with the best (pose + angle + face + body) adds ~+0.05 fitness, more than any
wording change. The server has everything for that score: RTMPose (`DetectScenePose`) on each result against the
mannequin's joints, and SSIM of the face/torso against the crop. Costs one extra generation (~2x time).

## Putting it in the app

1. `media-server/tasks/scene_finish.go`: `sceneFinishGrounding[FinishRepose]` = the text above with `{guide}` for
   `<image2>`, `{pose}` filled by `composeFinish` (its sentence dropped when empty, like `{target}`).
2. Pose words: port `describe` from `posewords.py` to Go; compute it in `GenerateLayerMask` from the paint layer's live
   `pose` ops (`paintIsPose`) in frame px, mapped into the crop (sides are relative to the picture).
3. Identity ref: in `GenerateLayerMask`, for a repose finish append the input crop as one more ref (it becomes
   `<image3>` after the paint guide); the grounding names it.
4. Mirror in `src/renderer/components/scene/scene-api.ts` `FINISH_GROUNDING.repose` (preview text) and update
   `TestSceneReposeGroundingAvoidsDrawingWords` / `TestSceneFinishPrompt`. None of the banned words appear.
5. Confirm at app settings (full crop size, 25 steps) on the user's two real scenes before shipping.

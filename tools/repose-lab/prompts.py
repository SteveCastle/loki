"""Every strategy tried: name -> (prompt for a case, pose reference rep, input).
rep: black | white | stick | capsule | pair | warp (reps.py). input: 'src' edits the picture with the rep as <image2>;
'warp' edits the cut-and-paste collage with the picture as <image2>. See SEARCH_SPACE.md."""
from posewords import describe


BASE = ("Make the person in <image1> do the exact same pose as shown in <image2>. Match the positions of the head, arms, hands and legs. "
        "Keep the person exactly as they are in <image1>: the same face, hair, clothing, body and style. "
        "Keep the background, the lighting, the camera angle and the framing of <image1> unchanged, and fill in naturally whatever the new pose uncovers.")

V2 = ("Make the person in <image1> do the exact same pose as shown in <image2>. <image2> only shows the pose: where the head, shoulders, elbows, hands, hips, knees and feet go. "
      "Follow the direction each part of the body points in <image2>, but keep this person's own body: the same proportions, limb lengths, build and size in the frame as in <image1>, so it is clearly the same body, only moved. "
      "They have exactly two arms, two hands and two legs, each joined naturally to the body; an arm or leg that moves leaves its old place, which then shows the background or their clothing instead. "
      "Keep their identity exactly: the same face, hair, skin and figure, and the same clothing, which drapes and folds naturally with the new pose. "
      "Keep the background, the lighting, the camera angle and the framing of <image1> unchanged.")

V3 = ("Show the same person from <image1> in a new pose: the exact same pose as shown in <image2>. {words}"
      "<image2> only marks where the head, shoulders, elbows, hands, hips, knees and feet go: follow the direction each part of the body points in it, not its lengths. "
      "It must be unmistakably the same person, only moved. Keep their exact face and expression, hair and skin, and their exact body type: "
      "the same weight, build, musculature, body fat, waist, hips, bust, thighs and curves, and the same limb lengths and proportions; "
      "do not slim, enlarge, smooth or idealise any part of the body. "
      "They have exactly two arms, two hands and two legs, each joined naturally to the body. An arm, hand or leg that moves leaves its old place completely, "
      "with nothing of it left behind, and the background or clothing shows there instead. "
      "The clothing stays the same and drapes naturally with the new pose. "
      "Keep the background, the lighting, the camera angle and the framing of <image1> unchanged.")

def auto(c):
    d = describe(c['target'], c['size'])
    return f"In the new pose: {d}. " if d else ''

def hand(c):
    return f"In the new pose: {c['words']}. "

V2W = lambda words: (lambda c: V2.replace('<image2> only shows', words(c) + '<image2> only shows'))

STRATS = {
    # generation 1 survivors and controls, re-run at 640 / low steps
    'base': (lambda c: BASE, 'black', 'src'),
    'v2w': (V2W(hand), 'black', 'src'),
    'v2auto': (V2W(auto), 'black', 'src'),
    # representation axis, v2auto wording
    'v2auto-white': (V2W(auto), 'white', 'src'),
    'v2auto-stick': (V2W(auto), 'stick', 'src'),
    'v2auto-capsule': (lambda c: V2W(auto)(c).replace('<image2> only shows the pose', '<image2> is a plain grey figure that only shows the pose'), 'capsule', 'src'),
    'v2auto-pair': (lambda c: V2W(auto)(c).replace('<image2> only shows the pose: where', '<image2> shows two figures: the pose they are in now on the left, and the new pose on the right. Use the one on the right: it shows where'), 'pair', 'src'),
    'warpref': (lambda c: auto(c) + "Make the person in <image1> do the exact same pose as the rough mock-up in <image2>, which shows this same person cut and pasted into the new pose. Follow its pose, not its seams, blur or gaps. Keep their exact face, hair, skin, body type, curves and clothing, the background, the lighting and the framing of <image1>.", 'warp', 'src'),
    'warpbase': (lambda c: "<image1> is a rough cut-and-paste mock-up of the person in <image2> moved into a new pose. " + auto(c) + "Turn it into a clean, natural picture of that same person in exactly this pose: rebuild every arm, hand and leg so it is whole and joined naturally to the body, remove every seam, smear and gap, and fill in the background where limbs used to be, matching <image2>. Keep the pose of <image1>, and keep the face, hair, skin, body type, curves and clothing of <image2> exactly.", 'warp', 'warp'),
}

STRATS.update({
    'v2': (lambda c: V2, 'black', 'src'),
    'v3': (lambda c: V3.replace('{words}', ''), 'black', 'src'),
    'v3auto': (lambda c: V3.replace('{words}', auto(c)), 'black', 'src'),
})

# ---- generation 3: mutations / crossovers of v2auto, plus new ideas
BODY = ("Keep their exact body type: the same weight, build, musculature, body fat, waist, hips, bust, thighs and curves. ")

def v2a(c):
    return V2W(auto)(c)

STRATS.update({
    # crossover v2auto x v3: v3's body-type list without its "do not slim..." brake
    'v4': (lambda c: v2a(c).replace('Keep their identity exactly:', BODY + 'Keep their identity exactly:'), 'black', 'src'),
    # the pose words lead the instruction
    'v2lead': (lambda c: (auto(c).replace('In the new pose: ', 'New pose for the person in <image1>: ') if auto(c) else '') + V2, 'black', 'src'),
    # words only, no figure: is the reference earning its place?
    'v2words-noref': (lambda c: (auto(c) + V2).replace('do the exact same pose as shown in <image2>', 'take exactly this new pose').replace(
        "<image2> only shows the pose: where the head, shoulders, elbows, hands, hips, knees and feet go. Follow the direction each part of the body points in <image2>, but keep",
        'Keep'), 'none', 'src'),
    # thicker figure lines (more signal per limb)
    'v2auto-thick': (lambda c: v2a(c), 'thick', 'src'),
    # tell it plainly that unmoved limbs stay and moved ones leave
    'v2only': (lambda c: v2a(c).replace('They have exactly two arms',
        'Only the arms and legs that are placed differently in <image2> move; the rest of the body stays exactly where it is. They have exactly two arms'), 'black', 'src'),
})

# ---- generation 4: cross the gen 3 leaders (v2lead, v2only, v4) + new ideas
ONLY = 'Only the arms and legs that are placed differently in <image2> move; the rest of the body stays exactly where it is. '

def lead(c, only=None):
    d = describe(c['target'], c['size'], only)
    return f"New pose for the person in <image1>: {d}. " if d else ''

def delta(c):
    d = describe(c['target'], c['size'], c['moved'])
    return f"Change only this: {d}. " if d else ''

X1 = lambda c: lead(c) + V2.replace('They have exactly two arms', ONLY + 'They have exactly two arms')
STRATS.update({
    'x1': (X1, 'black', 'src'),                                                     # v2lead x v2only
    'x2': (lambda c: X1(c).replace('Keep their identity exactly:', BODY + 'Keep their identity exactly:'), 'black', 'src'),  # x1 x v4
    'x1delta': (lambda c: delta(c) + V2.replace('They have exactly two arms', ONLY + 'They have exactly two arms'), 'black', 'src'),  # words for moved limbs only
    'x1loose': (lambda c: X1(c).replace('do the exact same pose as shown in', 'take the pose shown in'), 'black', 'src'),  # less literal
    'x1idref': (lambda c: X1(c).replace('Keep their identity exactly:', '<image3> is the same picture again, for their face and body. Keep their identity exactly:'), 'black+src', 'src'),
})

# ---- generation 5: sampling and representation on the x1 / v2lead line (wording has plateaued)
STRATS.update({
    'x1-ocapsule': (X1, 'ocapsule', 'src'),
})

# ---- generation 6: x1delta (best fidelity) line
FACING = 'Their head and body keep facing exactly the same way as in <image1>, at the same angle to the camera: only the arms and legs move. '
XD = lambda c: delta(c) + V2.replace('They have exactly two arms', ONLY + 'They have exactly two arms')
STRATS.update({
    'xd-face': (lambda c: XD(c).replace('They have exactly two arms', FACING + 'They have exactly two arms'), 'black', 'src'),
    'xd-lead': (lambda c: (delta(c).replace('Change only this: ', 'New pose for the person in <image1>, changing only this: ') ) + V2.replace('They have exactly two arms', ONLY + 'They have exactly two arms'), 'black', 'src'),
    'xd-face-body': (lambda c: XD(c).replace('They have exactly two arms', FACING + 'They have exactly two arms').replace('Keep their identity exactly:', BODY + 'Keep their identity exactly:'), 'black', 'src'),
})

# ---- generation 7: identity reference x delta words (the two fidelity winners), and a tight identity crop
IDREF = '<image3> is the same picture again, for their face and body. '
IDCROP = '<image3> is a close view of this same person, for their face and body. '
XDL = lambda c: (delta(c).replace('Change only this: ', 'New pose for the person in <image1>, changing only this: ')) + V2.replace('They have exactly two arms', ONLY + 'They have exactly two arms')
STRATS.update({
    'xdl-idref': (lambda c: XDL(c).replace('Keep their identity exactly:', IDREF + 'Keep their identity exactly:'), 'black+src', 'src'),
    'x1-idcrop': (lambda c: X1(c).replace('Keep their identity exactly:', IDCROP + 'Keep their identity exactly:'), 'black+crop', 'src'),
    'xdl-idcrop': (lambda c: XDL(c).replace('Keep their identity exactly:', IDCROP + 'Keep their identity exactly:'), 'black+crop', 'src'),
})

# ---- generation 8: per-image resolution (engine --jobs "sizes": [image1, image2, image3]; 0 = automatic)
STRATS.update({
    'x1-in1024': (X1, 'black', 'src', [1024, 0]),                 # the picture seen at 1024, figure as is
    'x1-fig384': (X1, 'black', 'src', [0, 384]),                  # the figure seen small
    'x1-in1024-fig384': (X1, 'black', 'src', [1024, 384]),
    'xdl-idref-hi': (STRATS['xdl-idref'][0], 'black+src', 'src', [0, 0, 1024]),   # identity ref seen at 1024
    'xdl-in1024-fig384': (XDL, 'black', 'src', [1024, 384]),
})

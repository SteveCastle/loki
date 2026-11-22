# MiniMax H3 (reference-to-video with sound)
<!-- sources: https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md, https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md -->
MiniMax H3 generates a video and its synchronized audio (speech, ambience, music) from text plus reference pictures, videos and audio. It is trained on long, explicit, shot-by-shot descriptions written in a fixed six-section format, and it is very sensitive to wording. Rewrite the user's request into that format. Do not write a plot summary or a bare list of references.

## Structure
Write six plain-text sections in this order, each starting with its name and a colon:
1. `subject_definitions:` one line for each reusable item, e.g. "<Subject 1> is the young woman in <Picture 1>, with long dark hair and a blue cardigan." Say what the label denotes, what it is used for, and the features to keep.
2. `summary:` one short paragraph that opens with the task type in brackets. Types are `keyframe completion`, `reference generation`, `video editing`, `video continuation`, `audio reuse` and `audio reference`, combined with " + ", e.g. `[reference generation + audio reference]`.
3. `retention_analysis:` one line per label, e.g. "<Subject 1> (appears in [Shot 1]): fully_preserved - …". The markers for visible content are `fully_preserved`, `partially_preserved`, `attribute_transfer` and `weak_reference`. The markers for audio are `fully_copy`, `partially_copy`, `reference` and `weak_reference`.
4. `detailed_description:` the main body. Open with one or two sentences of style and lighting. Then write `[Shot 1] …` with no timestamp, and later cuts as `[Shot 2] At 00:04.500, the shot cuts to …`. For each shot, cover composition, subject appearance and position, environment, light, actions, camera movement and the sound you hear.
5. `overall_soundscape:` 1-4 sentences on ambience and physical sounds across the whole video.
6. `non_diegetic_music:` 1-3 sentences on score that only the audience hears: instruments, tempo, dynamics. Write `N/A` if there is none.

## Rules
- Reference labels must match the official guide exactly. `<Picture N>` is a reference image used as a concrete frame or anchor ("the shot begins from <Picture 1>", "the shot ends on <Picture 2>"). `<Video N>` is a source video that is edited or continued, or whose cuts or rhythm are borrowed. `<Audio N>` is an audio track that is copied or referenced. `<Subject N>` is reusable visible content taken from those assets. Each label keeps one meaning everywhere. Each type is numbered on its own, in input order. Never invent references that were not supplied.
- With no references, drop the label sections and write only the description, soundscape and music.
- For generation, aim for 350-500 words in `detailed_description`. Every action must fit the clip length, roughly 5-15 s.
- Write camera motion as natural English: type + amplitude + speed. Types are push in/pull out, pan, truck, tilt, pedestal, arc shot, tracking shot, static shot, zoom, slight shake and POV. Example: "the camera pushes in with small amplitude at slow speed". Use a cut only when it brings new information.
- Speakers get stable IDs in the order they first speak: `(S1)`, `(S2)`. A referenced subject who speaks is written `<Subject 2> (S1)`. Dialogue goes in `<d>[English] Exact words.</d>`. Keep the speaker's own language, and end each line with . ? or !. For voiceover, write "says in an off-screen voiceover" and add that the on-screen lips stay closed. Use `<scenetrans>` when a line crosses a cut and `<cutoff>` when the video ends mid-line. Add dialogue only if the user asked for speech.
- Put visible on-screen text in double quotes, verbatim.
- Write all prose in English. Only dialogue, lyrics and on-screen text keep their own language.
- H3 has no negative prompt. Describe what is present instead.
- Keep the user's subjects and actions exactly. Never add people, nudity, brands, dialogue or on-screen text the user did not ask for.

## Example
Input: <Picture 1> my cat, make it yawn and stretch on the windowsill
Output:
subject_definitions:
<Subject 1> is the orange tabby cat in <Picture 1>, with a white chest, green eyes and a striped tail, lying on a sunlit windowsill.
<Picture 1> is the first frame of [Shot 1].

summary:
[keyframe completion] The target video begins from <Picture 1> and shows <Subject 1> waking, yawning widely and stretching along the windowsill in one continuous shot.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the cat's orange striped coat, white chest, green eyes and body proportions are retained.
<Picture 1> ([Shot 1] first frame): fully_preserved - the composition, windowsill, light and framing are retained at the opening.

detailed_description:
The target video is a live-action, naturalistic home video with warm late-afternoon sunlight and soft shadows.
[Shot 1] The shot begins from <Picture 1>: a medium close-up of <Subject 1> curled on a white-painted windowsill, warm sunlight falling across its fur from the window on the right. The cat's ears twitch, and it lifts its head and opens its eyes slowly. It opens its mouth in a wide, slow yawn that shows its pink tongue and small teeth, and its whiskers fan forward. It then rises onto its front legs and pushes its paws forward along the sill, back arched low and tail raised. Its claws spread briefly against the painted wood. The camera holds a static shot, then pushes in with small amplitude at slow speed as the cat finishes the stretch and settles back down, blinking toward the window. Dust motes drift through the sunbeam behind it.

overall_soundscape:
A soft, breathy squeak accompanies the yawn, followed by the faint scratch of claws on painted wood. Quiet indoor room tone and distant birdsong come through the window.

non_diegetic_music:
N/A

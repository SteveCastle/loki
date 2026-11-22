# Anima
<!-- sources: https://huggingface.co/circlestone-labs/Anima (README.md) -->
Anima (CircleStone Labs × Comfy Org) is a 2B anime and illustration text-to-image model built on Cosmos-Predict2, with a Qwen3-0.6B text encoder. It is trained on Danbooru-style tags, natural-language captions, and mixes of the two. It is made for anime and non-photorealistic art and does not do realism well. The best prompts lead with a quality prefix, then tags in the trained order, optionally followed by descriptive sentences.

## Structure
Tag order from the model card. Within a section, the order is free:
1. Quality / meta / year / safety: `masterpiece, best quality, score_7, safe,` and optionally `highres`, `absurdres`, `newest` or `year 2025`.
2. Subject count: `1girl`, `1boy`, `2girls`, `1other`, `no humans`.
3. Character: the character's name, if the user named one.
4. Series: the series name, if known.
5. Artist: `@artist name`. The @ is mandatory, or the effect is very weak.
6. General tags: appearance, clothing, pose, expression, setting, composition, lighting.
7. Optional natural language after the tags: two or more descriptive sentences, e.g. "masterpiece, best quality, 1girl. An anime girl with medium-length blonde hair is …".

## Rules
- Tags are lowercase, with spaces instead of underscores. Score tags (`score_7`) are the only tags that keep underscores. Prefer Gelbooru spellings where Danbooru and Gelbooru differ.
- In natural language, use normal capitalization for character and series names. Name a character, then describe their basic look. This matters most when there are several characters.
- Pure natural language needs at least two descriptive sentences. Very short prompts give unpredictable results.
- The model was trained with tag dropout, so cover the important visual facts rather than every possible tag. Typical length is 15-40 tags, or tags plus 2-4 sentences.
- Always include a safety tag (`safe` or `sensitive`) that matches the user's request. Never escalate it.
- Text rendering is weak: single words or short phrases at most.
- Weighting works but needs strong values, e.g. `(chibi:2)`.
- Negative prompts are supported (CFG 4-5; Turbo runs at CFG 1). Recommended negative: "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration". For the Aesthetic version, leave score tags out of both prompts.
- Never add artist tags, characters, people, nudity, brands or text the user did not ask for.

## Example
Input: anime girl reading under a cherry tree, spring
Output: masterpiece, best quality, score_7, safe, newest, 1girl, solo, sitting, under tree, cherry blossoms, falling petals, holding book, open book, reading, long hair, brown hair, school uniform, serafuku, pleated skirt, looking down, gentle smile, grass, spring, day, dappled sunlight, depth of field. A girl sits against the trunk of a large cherry tree in full bloom, reading a book in her lap while pink petals drift down around her. Soft afternoon sunlight filters through the blossoms and casts dappled light across her hair and the grass.

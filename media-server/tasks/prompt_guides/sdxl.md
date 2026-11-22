# Stable Diffusion XL
<!-- sources: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 (model card: dual CLIP encoders, limitations), https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/tokenizer/tokenizer_config.json (77-token context); no official prompt guide exists -->
SDXL is a latent diffusion model conditioned by two CLIP text encoders (CLIP ViT-L and OpenCLIP ViT-bigG). CLIP reads short phrases and keywords well and long grammatical sentences poorly, and its context is 77 tokens, about 75 usable. Longer prompts are split into 75-token chunks and each chunk is encoded on its own, so put the essentials in the first chunk. Write a compact, comma-separated list of descriptive phrases, most important first.

## Structure
1. Medium and style: "cinematic photo", "oil painting", "anime illustration", "3d render".
2. The main subject with its key attributes and action, as a short phrase ("a red fox sitting on a mossy log").
3. Setting and background.
4. Composition and camera: shot size, angle, lens, depth of field.
5. Lighting and colour: "golden hour", "soft rim light", "muted teal palette".
6. Texture and finish: "film grain", "visible brushstrokes", "highly detailed".

## Rules
- Use comma-separated phrases, not paragraphs. Keep the whole prompt within about 75 tokens (roughly 40-60 words) when possible. Never let the subject fall past the first chunk.
- Put the most important concepts first. Early tokens carry more weight.
- Emphasis follows ComfyUI syntax: `(phrase)` boosts ×1.1, `(phrase:1.3)` sets a weight. Keep weights between 1.1 and 1.4, and only on the few concepts that must win. Write `\(` and `\)` for literal parentheses.
- SDXL cannot render legible text reliably (model card limitation). Use at most one or two short words in quotes, and only if the user asked for text.
- Keep the composition simple. SDXL struggles with complex spatial relations ("a red cube on top of a blue sphere") and with several distinct people, so state relations plainly and give each subject its own attributes.
- Community fine-tunes (Pony, Illustrious, NoobAI) expect Danbooru tags and their own quality prefixes. If the checkpoint is a booru-style anime model, switch to lowercase tags (`1girl, solo, …`).
- Negative prompts are supported and useful at the default CFG (about 6; distilled checkpoints run at CFG 1 with no negative). The rewrite here is the positive prompt only. Do not write "no X" in it; mentioning X tends to add it.
- Never add people, nudity, brands or text the user did not ask for.

## Example
Input: cyberpunk street food stall at night
Output: cinematic photo, cyberpunk street food stall at night, steaming noodle bowls on a cluttered counter, narrow rain-soaked alley, glowing neon signs in pink and cyan, wet asphalt reflections, hanging paper lanterns, wide angle, low angle, shallow depth of field, volumetric haze, moody contrast, film grain, highly detailed

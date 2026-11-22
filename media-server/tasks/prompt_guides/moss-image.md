# MOSS-Image
<!-- sources: https://github.com/OpenMOSS/MOSS-Image/blob/master/README.md, https://github.com/OpenMOSS/MOSS-Image/blob/master/USERGUIDE.md, https://github.com/OpenMOSS/MOSS-Image/blob/master/skills/moss-image-prompt-enhancer/SKILL.md -->
MOSS-Image (OpenMOSS) is a 5B single-stream DiT with a Qwen3.5 text encoder. It accepts natural language, tag sequences and JSON layouts, and prompts of up to 4K tokens. It is strong at text rendering and handles photorealism and anime well. English has the most training data. The official prompt enhancer writes a detailed caption: cohesive prose that describes the visible choices, with spatial relations in words and any visible wording quoted exactly.

## Structure
1. The subject and its action, with the appearance that matters.
2. The setting and the spatial relations between elements (left/right, foreground/background, on, beside).
3. Camera and framing (close-up, wide, low angle, shallow depth of field) and the medium or style (photograph, oil painting, anime illustration, 3D render).
4. Lighting, materials, palette and mood.
5. Visible text, quoted exactly, with its placement, style and line breaks.

## Rules
- Write prose by default. The enhancer outputs a caption, not tags, and the caption may run to several paragraphs. Write no coordinates or JSON in it unless the user supplied a layout.
- Add detail where it makes the image more specific. Keep an explicitly minimalist brief minimalist. Do not use arbitrary word budgets: roughly 60-250 words is typical, longer for designed images with lots of text.
- Avoid redundant quality tags, unrequested style changes and invisible process notes (no "the AI should…", no "high quality").
- Text: keep visible wording verbatim in its original language, punctuation and capitalization, inside quotes. Use a real line break or `\n` where the user wants text to break across lines. Write the actual copy, never placeholders like "headline here". A quoted sentence in the request is not necessarily text to render, so decide which it is.
- For designed images (posters, cards), compose the information and imagery together: reading order, hierarchy, and text that is readable at the chosen size.
- Tag-style prompts also work. If the user writes tags (anime), keep tags and order them quality → score → meta → year → character → series → artist (with @) → general.
- A negative prompt is supported (CFG 3). The officially recommended one is "nsfw, worst quality, bad anatomy, watermark, jpeg artifact", but the rewrite itself is positive only.
- Preserve the user's subjects, counts, relationships, medium, style and exclusions. Never add people, nudity, brands or text the user did not ask for.

## Example
Input: a fox in a snowy forest at dawn, painterly
Output: A red fox stands alert on a snow-covered rise in a quiet pine forest at dawn, its head turned toward the viewer and one front paw lifted mid-step. Its thick russet coat is frosted with snow along the back, and its white chest and black-tipped ears stand out against the pale ground. Tall dark pines frame both sides of the scene, their branches bowed under heavy snow, and soft blue shadows fill the hollows between them. Behind the fox, the first low sunlight breaks through the trees in warm golden shafts, catching the drifting snowflakes and laying a peach-pink glow across the snow in the background. The image is a painterly digital illustration with visible, loose brushstrokes and soft edges, a cool palette of blues and violets set against warm gold and orange, and a calm, hushed morning mood.

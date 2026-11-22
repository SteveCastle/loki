# Qwen Image (text-to-image, Qwen Image 2.x)
<!-- sources: https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_t2i.txt, https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils_2512.py, https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils.py -->
Qwen Image is a text-to-image model with a Qwen-VL text encoder. It is especially strong at dense typography, posters and infographics, and Qwen Image 2.x accepts instructions of up to about 1k tokens. Qwen's official T2I prompt rewriter turns any request into one long English paragraph that describes the finished image as an observer would see it, walking the frame so that every element has a place.

## Structure
1. Opening sentence, about 20 words: medium noun (photograph, poster, illustration, infographic…) + style + subject + background or palette, usually with the orientation ("The image is a vertical watercolour illustration of …").
2. The background and the surface things sit on.
3. A walk through the frame. Either: top band → left / centre / right of the body → bottom band (for layouts and wide scenes); or, for one subject: background → pose and placement → head and face → each garment or surface → what it holds → the edges. Open about a third of the sentences with a positional phrase ("In the upper-left corner, …"). Aim for 8-14 such phrases that reach the corners and edges.
4. Every legible text string in reading order: where it sits, its weight, colour, case and size, and the words in double quotes.
5. A sentence of its own for the lighting: source, direction, quality, shadows and highlights.
6. A single closing sentence stepping back: "The overall composition …", covering balance, palette, style and mood.

## Rules
- Length: about 20 sentences / 400-500 words in the official 2.1 rewriter. Earlier Qwen-Image templates target about 200 words. A thin brief still gets a full description, but a simple scene stays restrained.
- Present tense, third person, declarative. Never "create" or "make sure". No quality boosters ("masterpiece", "8K").
- Name colours with a modifier (deep navy, warm terracotta) and give the material (brushed metal, coarse linen). Enumerate items. Write small counts as words.
- People get their observable surface only: build, posture, gaze, expression, hair, skin tone and each garment. Give age as a life stage ("in her thirties"), never a number of years.
- Objects by class, not brand, unless the user named one.
- Text: copy the user's strings character for character in straight double quotes, in their original script. Never invent signage when no text was requested.
- Turn negations into positive descriptions. Do not mention excluded objects at all.
- Do not put the aspect ratio or a resolution in the prompt.
- Qwen-Image's T2I example passes a quality-only negative prompt; Qwen Image 2.1's official examples run at CFG 1, where no negative prompt is used. The rewrite is positive only.
- Never add people, nudity, brands or text the user did not ask for.

## Example
Input: minimalist poster for a jazz festival called "Blue Hour Jazz"
Output: The image is a vertical minimalist graphic poster for a jazz festival, set on a smooth deep navy background with a faint paper grain. Across the top third, a large off-white serif headline reads "Blue Hour Jazz" in elegant upper and lower case, centred with generous letter spacing. In the centre of the frame, a single stylised tenor saxophone is drawn in flat brushed-gold shapes, angled diagonally from the lower left to the upper right, its bell opening toward the upper-right corner. Thin concentric arcs in pale powder blue radiate from the bell like sound waves, fading as they reach the right edge. On the left side, a narrow vertical band of muted teal runs from top to bottom, balancing the diagonal of the instrument. In the lower third, a solid circle in soft dusty orange sits behind the saxophone's keys, suggesting a low setting sun. Along the bottom edge there is open negative space with no other text. The lighting is flat and even, as in print design, with a subtle soft drop shadow beneath the saxophone that lifts it from the background. The overall composition is calm and asymmetrical, using a restrained palette of navy, gold, teal and orange and a quiet, late-evening mood.

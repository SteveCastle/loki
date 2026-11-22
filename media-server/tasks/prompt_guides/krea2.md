# Krea 2
<!-- sources: https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md, https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt, https://github.com/krea-ai/krea-2/blob/main/README.md -->
Krea 2 is Krea's open-weights text-to-image model, trained from scratch for creative and stylistic exploration. It comes as Raw (base) and Turbo (distilled, 8 steps, up to 2K). Krea recommends natural-language prompts and says long, detailed prompts give the best results. Krea's official prompt-expansion system prompt asks for a single cohesive paragraph that stays faithful to the user and is easy for the model to parse.

## Structure
One paragraph, typically:
1. The medium and style up front when it defines the image ("stylized digital painting of …", "A macro photograph of …", "1990s vintage anime style cel animation, …"), or the subject first for photographic scenes.
2. The subject(s), each grouped with its own attributes and action, in grounded phrasing for pose and interaction.
3. The setting and spatial layout (foreground, left/right, background).
4. Composition and framing: angle, shot size, depth of field.
5. Lighting, palette and texture (brushstrokes, film grain, paper texture), then the overall aesthetic.

## Rules
- Faithfulness first. Keep every original subject, action, colour and spatial relationship. Do not add objects, props, characters or animals the user does not clearly imply.
- Avoid over-specification. Do not invent highly specific clothing, colours or materials the input does not support. Choose the style, medium, framing and lighting yourself, but do not output your reasoning.
- If the prompt is already detailed, polish it lightly and keep the user's phrasing.
- Honor an explicitly requested medium ("photo of", "illustration of", "3D render of"). Never switch it.
- Official examples range from short phrases to about 200 words, and both prose sentences and dense comma-separated descriptive phrases work. Aim for about 50-150 words.
- Text rendering: wrap the exact words in quotes.
- One cohesive paragraph: no bullets, JSON, markdown or tags.
- Negative prompts: none are documented. Turbo runs with CFG disabled, so write everything positively.
- Treat people with dignity. Never add people, nudity, brands or text the user did not ask for.

## Example
Input: robot gardener, retro sci-fi illustration
Output: A retro science-fiction illustration of a friendly round-bodied robot gardener tending a bed of oversized tulips in a sunlit greenhouse. The robot has a polished cream-and-teal enamel shell, a domed glass head with two glowing amber eyes, and slender jointed arms, one holding a small brass watering can that pours a thin arc of water onto the flowers. Rows of terracotta pots and climbing vines fill the background beneath tall arched glass panes. Warm morning light streams through the glass, casting long geometric shadows across the tiled floor. The scene is rendered in a 1960s pulp magazine style with flat gouache colours, soft airbrushed gradients, a fine halftone texture, and a cheerful palette of teal, coral and butter yellow.

# MiniMax H3 Image
<!-- sources: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/raw/main/README.md, https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE (prompt/t2i/task_prompt.txt, prompt/edit/task_prompt.txt), https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/main/examples (revised_prompt.t2i.txt, revised_prompt.edit.txt) -->
MiniMax H3 Image is a distilled text-to-image and multi-reference image model that works natively at 4-16 MP. Its README states that it "expects a detailed English description of the whole image". MiniMax's official prompt expander turns a short request into one long, fluent English caption paragraph. Act like an expert visual designer, not a word expander: keep every hard fact from the input, then fill the image with concrete, visible content.

## Structure
Text-to-image (no references): one paragraph.
1. The image type and medium ("The image is an overhead, top-down photograph of …").
2. The main subjects with identity, pose, contact and scale.
3. The setting, walked region by region with explicit positions (upper left, lower center, on the right side).
4. Any visible text, placed on its surface.
5. Light: source, direction, quality and shadows, then lens or medium and the overall atmosphere.

With reference pictures (edit or composition): one paragraph in three parts.
1. Definitions: "The <descriptor> in <Picture 1> is <Subject 1>." Define every source element that stays visible or is transferred.
2. One sentence starting with "The target image …" that gives an overview, then one relationship clause per subject: "<Subject 1> is fully preserved.", "<Subject 2> is partially preserved; its identity and clothing remain, while its pose changes to …", or attribute transfer / weak reference.
3. Changed and new details only, described richly: new objects, changed attributes, newly exposed regions, and their shadows, contact and lighting. Never re-describe a fully preserved subject.

## Rules
- Write in English. Keep text in the image verbatim in its own script and language. For T2I, put that text in double quotes. In reference mode, the official expander quotes it in single quotes.
- References are `<Picture 1>`, `<Picture 2>`, … in upload order. Name each picture on its own and never write "the reference images".
- A long prompt is fine. The model reads up to 4096 tokens, and official expansions run about 150-350 words. Make the paragraph long because it is specific, not padded.
- State the subject's orientation relative to the camera (front-facing, left-facing profile, rear three-quarter). Say which hand touches what.
- Replace mood adjectives with visible cues. No quality tags ("8K", "masterpiece").
- Keep the requested subject count. Background people only if the user implies them.
- There is no negative prompt (it is rejected; the model runs at CFG 1). Describe what is present, not what is absent.
- Do not write aspect ratios or pixel sizes into the prompt.
- Never add people, nudity, brands or text the user did not ask for.

## Example
Input: cozy reading nook, rainy day
Output: The image is a warm, eye-level interior photograph of a cozy reading nook built into a bay window on a rainy afternoon. In the center of the frame, a deep cushioned window seat upholstered in faded sage-green linen holds a rumpled cream wool throw, two plump patterned pillows in mustard and rust, and an open hardcover book lying face down beside a white ceramic mug of tea with a thin curl of steam. Beyond the three tall window panes, heavy rain streaks down the glass in beaded rivulets, and the soft-focus garden outside shows dark wet hedges and a grey, overcast sky. On the left side, a tall built-in oak bookshelf is packed with books of mixed heights and worn spines, a small trailing pothos spilling over the top shelf. On the right, a brass floor lamp with a pleated fabric shade casts a warm amber pool of light across the cushions, and a stack of three paperbacks sits on the wide-plank floor at the lower right. The lighting mixes cool, diffused daylight from the window with the warm tungsten glow of the lamp, creating soft shadows in the folds of the throw and gentle highlights on the mug's glaze. The overall mood is quiet and sheltered, with shallow depth of field rendering the rain-covered glass slightly soft behind the crisp textures of linen, wool and wood.

# Qwen Image Edit (Qwen Image 2.x)
<!-- sources: https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_edit.txt, https://github.com/QwenLM/Qwen-Image-2.1/blob/main/README.md, https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils.py (polish_edit_prompt) -->
Qwen Image 2.x edits one or more input images from a text instruction. Qwen's official edit prompt rewriter turns a vague instruction into a precise, unambiguous editing directive anchored on what the input images show. Its governing principle is attribute disentanglement at full strength: change exactly what the user named, push the change to a clear and unmistakable degree, and keep everything else as it is in the input.

## Structure
One continuous paragraph, written as an instruction:
1. Lead with the operation verb: Replace / Add / Remove / Change / Turn / Place / Extend.
2. Name the exact target, with its location and the visible cues that set it apart.
3. Describe the new state concretely (category, colour, material, size, position, orientation). For a style, give its key visual features.
4. Add any direct consequences the change needs: newly exposed areas, contact shadows, reflections, matching light and perspective.
5. One blanket preservation clause naming what stays by type and role ("keep the person's face, pose and clothing and the rest of the scene unchanged").

## Rules
- References: with two or more images, use `<image1>`, `<image2>`, … and never "the first image" or "picture 2". `<image1>` is the image being edited, the canvas. State each image's role and exactly what is taken from it ("the jacket from <image2>"). With a single image, the official rewriter refers to it naturally as "the image".
- Name what stays without repainting it. Detailed description of preserved content makes it drift. Describe appearance only for what changes.
- Identity is the hardest invariant. Keep faces, products and the rendering medium (photo, anime, sketch) unless the user targets them. When identity comes from a reference, point at that image instead of describing the features.
- Do only what was asked. Do not clean up unmentioned defects, overlays or clutter.
- Text edits: write the exact characters in English double quotes and keep their language and case, e.g. Replace "SALE" with "OPEN".
- Expression, makeup and beauty changes must be natural and subtle. When the user asks for outpainting, name it as outpainting.
- Be decisive. Resolve vagueness and contradictions into one concrete reading, with no hedging and no alternatives. State requirements affirmatively.
- Never write resolution, aspect ratio or "4K" into the prompt.
- Write the prose in English unless the user wrote in Chinese. The official rewriter keeps Chinese for Chinese instructions.
- There is no negative prompt (the official settings run CFG 1).
- Never add people, nudity, brands or text the user did not ask for.

## Example
Input: put the hat from image2 on the guy in image1
Output: Place the wide-brimmed straw hat from <image2> on the head of the man in <image1>, seated naturally on his hair with the brim level above his eyebrows and tilted slightly back, scaled to his head size and matching the angle of his head. Light the hat with the same warm side light that falls on his face in <image1>, with a soft shadow from the brim across his forehead and matching shadow on his hair. Keep the man's face, expression, hairline, clothing and pose, and the entire background of <image1>, exactly unchanged.

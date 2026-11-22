# FLUX.2 [klein]
<!-- sources: https://docs.bfl.ai/guides/prompting_guide_flux2_klein (archived 2026-04-20: http://web.archive.org/web/20260420011101/https://docs.bfl.ai/guides/prompting_guide_flux2_klein), https://docs.bfl.ml/guides/prompting_unified_reference.md, https://docs.bfl.ai/guides/prompting_guide_flux2.md -->
FLUX.2 [klein] (4B / 9B) is Black Forest Labs' small, fast FLUX.2 model with a Qwen3 text encoder. BFL's guidance is that it "works best when you describe scenes like a novelist, not a search engine". It has no prompt upsampling: what you write is exactly what the model receives. Turn the user's request into flowing, descriptive prose with explicit lighting.

## Structure
Subject → Setting → Details → Lighting → Atmosphere, with an optional style/mood tag at the end:
1. Subject and key action first. Word order signals priority.
2. Setting: where the scene takes place.
3. Details: specific visual elements, textures, materials and reflections.
4. Lighting, described like a photographer would: source (natural, artificial, ambient), quality (soft, harsh, diffused), direction (side, back, overhead, camera-left), temperature (warm, cool, golden) and how it interacts with surfaces.
5. Atmosphere and mood.
6. Optional: end with "Style: … Mood: …", or a camera or film line such as "Shot on 35mm film (Kodak Portra 400) with shallow depth of field", for a consistent look.

## Rules
- Write prose, never comma-separated keyword lists.
- Lighting has the single greatest impact on output quality. Always describe it specifically, never "good lighting".
- Length: 30-80 words suits most images. Use 80-300 words for complex editorial or product scenes, but every sentence must add visual information.
- Use sensory detail ("flaky croissant layers catching soft diffused light") rather than bare nouns.
- Text in the image: put the exact words in quotation marks and say where they sit and in what style. Keep text short.
- No negative prompts (FLUX.2 does not support them). Describe what you want, not what you don't want.
- English gives the most precise results.
- If reference images are attached (up to 4), refer to them as "image 1", "image 2". Describe what changes or how they combine, not what they look like.
- Never add people, nudity, brands or text the user did not ask for.

## Example
Input: old lighthouse on a cliff, stormy
Output: A weathered white lighthouse stands on the edge of a jagged black cliff as a storm rolls in over the sea. Its paint is peeling and streaked with rust, and the lamp at the top throws a pale beam out into the rain. Below, huge grey-green waves crash against the rocks and burst into white spray that drifts up the cliff face. Heavy slate-coloured clouds press low over the horizon, broken by a single thin gap where cold silver light breaks through and rims the lighthouse and the wet stone in a hard, metallic edge, leaving the foreground in deep blue shadow. Style: dramatic seascape photography, wide angle. Mood: wild, lonely, defiant.

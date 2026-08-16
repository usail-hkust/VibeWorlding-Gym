"""
prompts_eng.py — English versions of all verifier LLM prompt constants (English mirror of prompts.py).

Contains all prompts required by the three verifier families:
  1. unverified (refine) v2: HARD_SYSTEM_PROMPT (legacy monolithic), HARD_H12_SYSTEM_PROMPT, H3_VURR_SYSTEM_PROMPT, SOFT_*
  2. onestep_scene (generate): HARD_H12_SCENE_SYSTEM_PROMPT, H3_SCENE_SYSTEM_PROMPT, H5_SCENE_SYSTEM_PROMPT
  3. All matching USER_PROMPT counterparts.

Notes:
  - unverified v2 uses only HARD_H12_* + H3_VURR_* + SOFT_* (H3 is split into VU + VR as two independent calls).
  - HARD_SYSTEM_PROMPT (legacy H1+H2+H3 monolithic) is kept for backward compatibility; the new pipeline does not use it.
  - The deprecated 10-dimension offline aesthetic prompts
    (ECOLOGY/THEME/HEIGHT/STYLE/COLOR_HARMONY/COMPANION/ECO_LAYER/DENSITY/LAYOUT/DIVERSITY) have been removed.
"""

# ============================================================
# [unverified v1 legacy] HARD — monolithic H1+H2+H3 (kept for the old pipeline)
# ============================================================

HARD_SYSTEM_PROMPT = """
You are an expert in 3D scene quality evaluation. Please perform **Hard Constraints validation** on the following scene.
Judge whether the scene passes on three hard-constraint dimensions (pass/fail).

## Background
This evaluation only scores the "combination of AI-placed elements" itself. Ignore uncontrollable factors:
- Sky color (day/night/aurora background), background sea, water-surface lighting, and other render-backdrop issues.
Focus only on: element selection, placement position, ecological plausibility, and thematic consistency.

## Judgment principle (extremely important)
For each Hard dimension:
- **If any single problematic element is found -> the dimension fails immediately (pass=0).**
- **If no problematic element is found -> the dimension passes (pass=1).**
- Do not compute ratios or apply tolerances — any problem means fail.

## Three Hard constraints

### H1 - Height plausibility
Check every element's Z-axis placement:
- Ground objects (plants, buildings, furniture, etc.) should have their base touching the terrain or be reasonably embedded.
- No obvious floating (hovering in air) or clipping (sinking underground).
- Effect elements (light beams, particles, smoke, etc.) may float — not counted as a problem.
- Game scenes allow some height exaggeration, but placements that are physically absurd are not acceptable.

**Important exemptions (do NOT flag as H1 issues):**
- Elements at Z=0: Z=0 is the world-space ground baseline; anything at Z=0 is "standing on the ground" even if the render backdrop shows water/sea.
- Terrain elements (rocky mountains, natural stones, limestone, sand blocks, etc.): always plausible at Z>=0 — they define the terrain itself.
- Effect/particle elements (beams, glows, smoke, particles, stars, ripples, etc.): may appear at any height.
- Flying creatures (birds, bats, etc.): being high is normal flight, not floating.
- Underwater creatures (fish schools, water plants, jellyfish, etc.): Z<0 is underwater and plausible.
- The rendered water/sea backdrop is an engine backdrop, not a real water body. Do not conclude that ground elements are "floating on water" because of it.
- **Strictly consult the rule-based preflight results below.** Do not re-penalize elements the preflight has already cleared.
- For elements marked "has support analysis" in the preflight, the bounding-box check is only an approximate reference — **defer to the actual visual in the screenshots** when judging genuine floating.
- For range-generated elements (many placed in one call), the Z range only denotes the sampling interval; most individuals sit low with support. Do not flag them all as floating just because the Z upper bound is high.
- **Note: element collision/overlap issues are handled by an independent H4 dimension. H1 does not need to concern itself with clipping/overlap.**

**Judgment: any implausible-height element -> H1=0; all plausible -> H1=1.**

### H2 - Ecological plausibility
Check every element for ecological plausibility along these sub-aspects:
(a) Element–terrain match: does the element fit the terrain type? (No tropical plants on snowfields, etc.)
(b) Companion relations: do co-existing species come from the same or adjacent biomes?
(c) Ecological layering: is the canopy–shrub–groundcover stratification reasonable?
(d) Functional zoning: is the placement location reasonable? (Streetlamps belong beside roads, not in the middle of a lake.)

**Important caveats:**
- Fantasy / magic / cyberpunk / sci-fi themes should relax ecological standards substantially and permit supernatural combinations.
- If the user's query explicitly requests a style change (e.g., "turn it into a rainforest", "more tropical"), judge against the **target style** rather than the original terrain type.
- Elements already present in the initial scene (compare initial vs. final element lists) should not be penalized for ecological mismatch — that is the original design.
- Invisible elements (Z far below ground, not visible in the screenshots) should not participate in judgment.
- The "water" terrain type in game scenes refers to the rendered backdrop; plants/buildings on the Z=0 plane are normal placements, not "planted on water".

Note: fantasy/sci-fi themes warrant relaxed ecological standards.
**Judgment: any ecologically implausible element -> H2=0; all plausible -> H2=1.**

### H3 - Thematic consistency and instruction following
Check every element against the theme + description in the query:
- Elements that clearly contradict the theme -> problematic (e.g., a modern streetlamp in a pirate harbor).
- Neutral elements (neither contradicting nor reinforcing) -> not counted as problematic.

**Instruction-following check (important):**
- Look at the agent's final response and judge whether it addressed the user's request reasonably.
- If the user explicitly requested an action (e.g., "replace X with Y", "add Z") but the agent neither executed it nor explained why -> H3 issue.
- If the request exceeded the capability of the available assets (see the available-assets list) and the agent made a reasonable substitution OR explained the reason in the response -> **do not penalize**.
- If the available-assets list truly lacks the requested element and the agent said so in the response -> not an H3 issue.

**Important caveats:**
- Elements already in the initial scene (compare initial vs. final element lists) should not be judged as thematically inconsistent — they belong to the original design, not to this AI edit.
- In stylized game scenes (cartoon, fantasy, chibi), stylistic differences between elements have higher tolerance.
- Only flag elements that **clearly contradict** the theme, not merely "imperfectly matched" ones.
- In fantasy/magic themes, cross-style elements (fossils in a magic forest, classical stones, etc.) can be seen as adding mystique.
- **Assess against the available-assets list:** if the assets in component_info themselves are unsuitable for the requested theme, that is an asset-library limitation, not the agent's fault.

**Judgment: any element that clearly contradicts the theme, OR the agent failed to fulfill the user's core request without explanation -> H3=0; all plausible -> H3=1.**

## Evaluation procedure
1. Carefully inspect the screenshots and element list.
2. For H1/H2/H3, check every element.
3. Once a problematic element is found, list the specific issue and set that dimension to 0.
4. If no problem is found for a dimension, set it to 1.

## Output format (strict JSON, no extra text)
```json
{
  "H1": {
    "pass": 0 or 1,
    "issues": [
      {"element": "element name", "problem": "specific issue description"}
    ]
  },
  "H2": {
    "pass": 0 or 1,
    "issues": [
      {"element": "element name", "sub_aspect": "terrain_match / companion / layer / zone", "problem": "specific issue description"}
    ]
  },
  "H3": {
    "pass": 0 or 1,
    "issues": [
      {"element": "element name", "problem": "specific issue description"}
    ]
  },
  "hard_pass": true or false,
  "summary": "one-sentence summary"
}
```
Note: when `issues` is an empty list, `pass` must be 1; when `issues` is non-empty, `pass` must be 0.
""".strip()

HARD_USER_PROMPT = """
## Scene info
- Theme: {theme}
- User request: {query}
- Terrain type: {terrain_type}

## Initial scene element list
{init_map_summary}

## Final scene element list
{final_map_summary}

## Rule-based preflight results (H1 height preflight)
{rule_flagged_issues}

## Element knowledge base
{knowledge_base_context}

## Agent's final response (agent's user-facing reply)
{agent_final_response}

## Available assets (all element types the agent can use)
{available_assets}

## Screenshots
### Initial scene screenshots (before edit)
{init_image_tags}

### Final scene screenshots (after edit)
{final_image_tags}
""".strip()


# ============================================================
# [unverified v2] HARD H1/H2 — H3 has been split out into an independent VU+VR pipeline
# ============================================================

HARD_H12_SYSTEM_PROMPT = """
You are an expert in 3D scene quality evaluation. Please perform **Hard Constraints H1/H2 validation** on the following scene.
You only need to judge whether the scene passes on H1 (height plausibility) and H2 (ecological plausibility), the two hard-constraint dimensions (pass/fail).
H3 (user intent / need satisfaction) is handled by an independent VU+VR pipeline — **do NOT evaluate H3 in this call**.

## Background
This evaluation only scores the "combination of AI-placed elements" itself. Ignore uncontrollable factors:
- Sky color (day/night/aurora background), background sea, water-surface lighting, and other render-backdrop issues.
Focus only on: element selection, placement position, ecological plausibility.

## Judgment principle (extremely important)
For each Hard dimension:
- **If any single problematic element is found -> the dimension fails immediately (pass=0).**
- **If no problematic element is found -> the dimension passes (pass=1).**
- Do not compute ratios or apply tolerances — any problem means fail.

## Two Hard constraints

### H1 - Height plausibility
Check every element's Z-axis placement:
- Ground objects (plants, buildings, furniture, etc.) should have their base touching the terrain or be reasonably embedded.
- No obvious floating (hovering in air) or clipping (sinking underground).
- Effect elements (light beams, particles, smoke, etc.) may float — not counted as a problem.
- Game scenes allow some height exaggeration, but placements that are physically absurd are not acceptable.

**Important exemptions (do NOT flag as H1 issues):**
- Elements at Z=0: Z=0 is the world-space ground baseline; anything at Z=0 is "standing on the ground" even if the render backdrop shows water/sea.
- Terrain elements (rocky mountains, natural stones, limestone, sand blocks, etc.): always plausible at Z>=0.
- Effect/particle elements (beams, glows, smoke, particles, stars, ripples, etc.): may appear at any height.
- Flying creatures (birds, bats, etc.): being high is normal flight.
- Underwater creatures (fish schools, water plants, jellyfish, etc.): Z<0 is underwater and plausible.
- **Strictly consult the rule-based preflight results below.** Do not re-penalize elements the preflight has already cleared.
- For elements marked "has support analysis" in the preflight, the bounding-box check is only an approximate reference — **defer to the actual visual in the screenshots**.
- **Note: element collision/overlap issues are handled by an independent H4 dimension. H1 does not need to concern itself with clipping/overlap.**

**Judgment: any implausible-height element -> H1=0; all plausible -> H1=1.**

### H2 - Ecological plausibility
Check every element for ecological plausibility along these sub-aspects:
(a) Element–terrain match: does the element fit the terrain type?
(b) Companion relations: do co-existing species come from the same or adjacent biomes?
(c) Ecological layering: is the canopy–shrub–groundcover stratification reasonable?
(d) Functional zoning: is the placement location reasonable?

**Important caveats:**
- Fantasy / magic / cyberpunk / sci-fi themes should relax ecological standards substantially.
- If the user's query explicitly requests a style change, judge against the **target style** rather than the original terrain type.
- Elements already present in the initial scene (compare initial vs. final element lists) should not be penalized for ecological mismatch.
- Invisible elements should not participate in judgment.
- The "water" terrain type in game scenes refers to the rendered backdrop; plants/buildings on the Z=0 plane are normal placements.

**Judgment: any ecologically implausible element -> H2=0; all plausible -> H2=1.**

## Evaluation procedure
1. Carefully inspect the screenshots and element list.
2. For H1/H2, check every element.
3. Once a problematic element is found, list the specific issue and set that dimension to 0.
4. If no problem is found for a dimension, set it to 1.

## Output format (strict JSON, no extra text)
```json
{
  "H1": {
    "pass": 0 or 1,
    "issues": [
      {"element": "element name", "problem": "specific issue description"}
    ]
  },
  "H2": {
    "pass": 0 or 1,
    "issues": [
      {"element": "element name", "sub_aspect": "terrain_match / companion / layer / zone", "problem": "specific issue description"}
    ]
  },
  "summary": "one-sentence summary"
}
```
Note: when `issues` is an empty list, `pass` must be 1; when `issues` is non-empty, `pass` must be 0.
""".strip()


HARD_H12_USER_PROMPT = """
## Scene info
- Theme: {theme}
- User request: {query}
- Terrain type: {terrain_type}

## Initial scene element list
{init_map_summary}

## Final scene element list
{final_map_summary}

## Rule-based preflight results (H1 height preflight)
{rule_flagged_issues}

## Element knowledge base
{knowledge_base_context}

## Available assets (all element types the agent can use)
{available_assets}

## Screenshots
### Initial scene screenshots (before edit)
{init_image_tags}

### Final scene screenshots (after edit)
{final_image_tags}
""".strip()


# ============================================================
# [unverified v2] H3 — H3-VU (Visual Understanding) + H3-VR (Visual Reasoning)
# H3_pass = VU >= 4 AND VR >= 4
# ============================================================

H3_VURR_SYSTEM_PROMPT = """
You are a quality-evaluation expert for 3D-scene-editing agents. Your task: given one complete agent conversation, score two sub-dimensions on a 0-5 scale, used to decide whether the user's intent / need was satisfied (H3).

## H3 recap
H3 = whether the user's intent / need is satisfied. It can fail in two stages:
  - [Visual Understanding stage] The agent failed to correctly understand the scene / did not identify the implicit intent in the user's query -> H3-VU issue.
  - [Visual Reasoning stage]     The agent understood correctly, but the tool_calls were wrong / the final map does not satisfy the request -> H3-VR issue.

You must give the two sub-dimensions **independent** scores, then produce the final H3_pass.

## H3-VU (Visual Understanding): Did the agent correctly understand the scene and identify the user's intent?

**Inputs:**
  - User query
  - Initial scene (map_json + scene image)
  - Agent's final-turn response (user-facing final reply — the primary focus)
  - Middle-turn thinking (as a reference to help judge the understanding process)

**Focus:**
  1. Is the agent's understanding of the current scene accurate?
     - Did it correctly identify element names, positions, and counts in the scene?
     - Did it correctly grasp spatial relationships between elements?
  2. Did the agent correctly identify the implicit intent(s) in the user's query?
     - Which element(s) to operate on? What kind of operation (add / delete / modify)?
     - In multi-intent queries, were any intents missed?
  3. If the user's request involves elements not supported by the asset catalog, did the agent recognize this and reasonably explain it in the response?
     (Recognized + explained -> no VU penalty.)

**Scoring rubric:**
  0: No response at all / cannot be evaluated.
  1: No intent recognized at all; scene understanding entirely wrong.
  2: Partial intent recognition with major omissions or misreadings; obvious errors in scene understanding.
  3: Main intent recognized, but secondary intents missed or scene details slightly off.
  4: Intent recognition essentially complete and accurate, scene understanding correct, only minor deviations.
  5: All operational intents identified perfectly, scene understanding flawless.

## H3-VR (Visual Reasoning): Did the agent make correct reasoning + tool calls based on that understanding?

**Inputs:**
  - User query
  - Initial scene map_json
  - Full ordered list of tool_calls across all turns
  - Final scene map_json
  - Agent's final-turn response

**Focus:**
  1. **Does the final map (final_map_json) satisfy the user's request?** (Core criterion)
     - Compare initial and final map_json — did the edits actually take effect?
     - Does the final result answer what the user asked for?
  2. **Is the intermediate tool_call chain reasonable?** (Auxiliary criterion — walk through them one by one)
     - Is each tool_call consistent with the intent the agent claimed to understand?
     - Are there "thinking says change, but tool_call doesn't change" cases (Q3b modification hallucination)?
     - Any "no tool call at all, just a text reply" cases (Q3c) -> VR <= 1 directly.
  3. If the user's request exceeds the asset library, and the agent made a reasonable substitution or explained -> no VR penalty.
  4. Are the tool_call parameters reasonable (position / count / orientation)?

**Scoring rubric:**
  0: No tool_call executed at all / no valid tool_call.
  1: tool_calls entirely inconsistent with understanding; final map does not satisfy the request.
  2: Some operations, but with major omissions/errors; final map partially satisfies.
  3: Main operations correct, final map largely satisfies, secondary omissions.
  4: Operations essentially complete and accurate, final map satisfies the request, only minor deviations.
  5: All operations perfectly accurate; final map perfectly satisfies the user's request.

## H3_pass rule
  - H3_pass = 1 iff (VU >= 4) AND (VR >= 4).
  - Otherwise H3_pass = 0.

## Output format (strict JSON, no extra text)
```json
{
  "H3_VU": {
    "score": integer 0-5,
    "evidence": "...(which intents were understood/missed, whether scene understanding is accurate)"
  },
  "H3_VR": {
    "score": integer 0-5,
    "evidence": "...(whether the final map satisfies the request, which tool_calls were correct/wrong)"
  },
  "H3_pass": 0 or 1,
  "H3_pass_reason": "VU=X, VR=Y, pass condition: VU>=4 AND VR>=4"
}
```
""".strip()


H3_VURR_USER_PROMPT = """
## Scene info
- Theme: {theme}
- User request (query): {query}
- Terrain type: {terrain_type}

## Initial scene map_json summary
{init_map_summary}

## Final scene map_json summary
{final_map_summary}

## Agent multi-turn transcript (in turn order)
Below is the thinking/content of every assistant turn plus its tool_calls.
The last assistant turn is the "final-turn response" returned to the user.

{agent_turns}

## Agent's final-turn response (user-facing)
{agent_final_response}

## Available assets (all element types the agent can use)
{available_assets}

## Screenshots
### Initial scene screenshots (before edit)
{init_image_tags}

### Final scene screenshots (after edit)
{final_image_tags}

## Task
Following the system prompt, score H3-VU and H3-VR independently (0-5), then produce H3_pass (VU>=4 AND VR>=4).
""".strip()


# ============================================================
# [DEPRECATED 2026-07-03] SOFT — S1/S2/S3 aesthetic evaluation
# Soft rewards have been removed from the whole pipeline (unverified reward = hard_pass ? 1.0 : 0.0).
# The prompt definitions below are retained only for historical compatibility; no code references them.
# ============================================================

SOFT_SYSTEM_PROMPT = """
You are an expert in 3D-scene aesthetic evaluation. The scene has already passed Hard Constraints (height plausible, ecologically plausible, thematically consistent);
now please score the scene from the perspective of **aesthetic quality**.

## Background
This evaluation only scores the "combination of AI-placed elements" itself. Ignore uncontrollable factors:
- Sky color (day/night/aurora background), background sea, water-surface lighting, and other render-backdrop issues.
Focus only on: visual style of elements, variety of kinds, and spatial density distribution.

## Three Soft dimensions (each 1-5)

### S1 - Visual style consistency
Overall judgment of whether the art style, material feel, and color palette of the elements harmonize.
Covers: consistency of style type (photoreal / cartoon / fantasy / ink-wash) + color harmony.
**Also check: whether asset postures look normal** — e.g., are trees severely tilted, buildings toppled, elements abnormally rotated so as to look visually unnatural. Serious posture anomalies should noticeably lower the S1 score.
- 5: Highly unified, palette harmonious, postures natural, professional design feel.
- 4: Overall coherent, occasional differences but not jarring, postures basically normal.
- 3: Obvious style mixing, palette somewhat off, or a few elements have abnormal postures.
- 2: Style chaotic, visually incoherent, or multiple elements have unnatural postures.
- 1: Completely chaotic, no recognizable unified style, many elements with abnormal postures.

### S2 - Element variety
Overall judgment of whether the scene uses a sufficient variety of element kinds and whether each layer varies.
**Also check: whether similar-kind elements show reasonable variation** — do same-type trees/rocks vary in rotation and size, or are they identical clones?
- 5: Rich variety, all layers (large/medium/small) vary, same-kind elements show reasonable differences, strong depth.
- 4: Fairly non-monotonous, layers show some variation.
- 3: Acceptable, but some layers are under-varied.
- 2: Obviously repetitive, many same-kind elements filled in with identical poses; visually monotonous.
- 1: Nearly only one kind of element; extremely monotonous.

### S3 - Density and layout plausibility
Overall judgment of the density distribution and layout quality of elements in space.
Covers: natural density + layered spatial distribution + presence of a visual focal point.
**Scoring requirements (strictly enforced):** A score of 2 means the layout has multiple obviously visible problems (grid-like arrangement, large piled-up or empty areas, lack of depth); do not casually go above 3. Only give 3+ when the layout genuinely achieves natural density and depth.
- 5: Well-varied density, natural distribution, balanced composition with room to breathe.
- 4: Basically reasonable, minor local deviations.
- 3: Layout is acceptable; some planning but not refined, parts too empty or too dense.
- 2: Clear layout problems, multiple messy or empty regions, no sense of depth, or elements arranged in a grid.
- 1: Serious density/layout problems — large piled-up regions or large empty regions.

## Scoring procedure
1. Given the screenshots and element list, observe the overall visual effect.
2. Score each dimension (S1/S2/S3 independently).
3. Give an integer score and short reason per dimension.

## Output format (strict JSON, no extra text)
```json
{
  "S1": {"score": X, "reason": "reason"},
  "S2": {"score": X, "reason": "reason"},
  "S3": {"score": X, "reason": "reason"},
  "soft_avg": value of (S1+S2+S3)/3,
  "summary": "one-sentence summary"
}
```
""".strip()

SOFT_USER_PROMPT = """
## Scene info
- Theme: {theme}
- User request: {query}
- Terrain type: {terrain_type}

## Final scene element list
{final_map_summary}

## Agent's final response (for reference: what edits did the agent make?)
{agent_final_response}

## Screenshots
The images below show the final scene from multiple angles:
{image_tags}
""".strip()


# ============================================================
# [onestep_scene] H1/H2 — from-scratch version (only height + ecology)
# ============================================================

HARD_H12_SCENE_SYSTEM_PROMPT = """
You are an expert in 3D scene quality evaluation. This is a "from-scratch" scene: given only one line of user text (the query),
the agent used retrieval tools to build the whole scene from nothing (no initial scene; every element is newly placed).
Judge only **H1 (height plausibility)** and **H2 (ecological / common-sense plausibility)** as pass/fail.
H3 (need satisfaction) and H5 (retrieval usage / style fit) are handled by independent pipelines — **do NOT evaluate H3/H5 here, and do NOT judge "whether the user's needs are met" or "whether the style matches the theme"**.

## Background
Only evaluate whether the "combination of AI-placed elements" is plausible in a **physical / ecological / common-sense** way. Ignore uncontrollable / out-of-scope factors:
- Sky color (day/night/aurora), background sea, water-surface lighting, and other render-backdrop issues.
- **Whether the user's need is met, whether theme elements are complete, whether the style fits** — these are NOT H2's concern; H3/H5 handle them.
Focus only on: element placement (Z axis, H1) and ecological / common-sense plausibility (H2).

## Judgment principle (extremely important)
- **Any single problematic element -> the dimension fails immediately (pass=0).**
- **No problematic elements -> the dimension passes (pass=1).**
- No ratios, no tolerance — any problem means fail.

## H1 - Height plausibility
Check every element's Z-axis placement:
- Ground objects (plants, buildings, furniture, props, etc.) should have their base touching the ground or be reasonably embedded, with no obvious floating or clipping.
- **Exemptions (NOT to be flagged as H1):**
  - Z=0 is the world ground baseline; any element at Z=0 counts as "standing on the ground" (even if the backdrop is rendered as water).
  - Terrain elements (stone slabs, ground blocks, rocky mountains, reefs, sand, etc.) are always plausible at Z>=0.
  - Effect/particle elements (beams, glows, smoke, flames, particles, ripples, etc.) may appear at any height.
  - Flying creatures at height, underwater creatures at Z<0 are all plausible.
  - **Strictly consult the rule-based height preflight below.** Do not re-penalize what the preflight has cleared; defer to the actual visual.
  - Collision/clipping/overlap is handled by H4 independently; H1 does not concern itself with it.

**Judgment: any implausible-height element -> H1=0; all plausible -> H1=1.**

## H2 - Ecological / common-sense plausibility
**Judge only objective ecological / physical common-sense plausibility. Do NOT judge "whether the user's need is met" or "whether the theme elements are complete"**
(whether elements answer named user needs and whether the style matches the mood are evaluated by H3 and H5; do not double-penalize in H2).
Check every element for **objective ecological or common-sense conflicts**:
(a) Element–environment physical/ecological conflict: tropical palms on snowy mountains, water plants in dry desert, deep-sea creatures on land, etc.
(b) Companion-relation absurdities: species impossible to co-exist in the same natural environment piled together.
(c) Functional-zoning common-sense errors: doors facing walls, staircases floating into thin air, etc.

**Important relaxations (NOT counted as H2 issues):**
- **"Approximate substitution" caused by the retrieval missing an asset is NOT an H2 issue** — that lives at the retrieval / need-satisfaction level, judged by H3/H5.
- **Style / period not perfectly matching the theme** (too cartoony, not gritty enough, wrong era feel) is NOT H2 — it's H5's concern.
- Fantasy / magic / cyberpunk / sci-fi / post-apocalyptic themes: sharply relax ecology; ecological conflicts almost never apply.
- Invisible elements do not participate in judgment.
- As long as the element sits in a location that is **physically / ecologically plausible**, it's fine.

**Judgment: any element objectively violating ecology / common sense -> H2=0; otherwise -> H2=1.**

## Output format (strict JSON, no extra text)
```json
{
  "H1": {"pass": 0 or 1, "issues": [{"element": "element name", "problem": "specific issue"}]},
  "H2": {"pass": 0 or 1, "issues": [{"element": "element name", "sub_aspect": "ecological_conflict / companion_absurdity / zoning_common_sense", "problem": "specific issue"}]},
  "summary": "one-sentence summary"
}
```
Note: empty `issues` -> pass=1; non-empty `issues` -> pass MUST be 0.
""".strip()


HARD_H12_SCENE_USER_PROMPT = """
## Scene info (from-scratch, no initial scene)
- Theme: {theme}
- User query: {query}
- Terrain hint: {terrain_type}

## Final scene element list (agent-built from scratch)
{final_map_summary}

## Rule-based height preflight (H1 reference)
{rule_flagged_issues}

## Element knowledge base
{knowledge_base_context}

## Final scene screenshots (5 views: left / right / front / back / top)
{final_image_tags}
""".strip()


# ============================================================
# [onestep_scene] H3 — VU + VR + Response (distractor recognition)
# H3_pass = VU >= 4 AND VR >= 4 AND Response_pass == 1
# ============================================================

H3_SCENE_SYSTEM_PROMPT = """
You are a quality-evaluation expert for 3D-scene-generation agents. This is a "from-scratch" task: given one query, the agent retrieves assets and builds the whole scene.
Evaluate "whether the user's needs are satisfied" (H3) along three sub-dimensions: H3-VU, H3-VR, and H3-Response.

## H3-VU (Visual Understanding, 0-5): Does the agent's thinking correctly grasp the "asset / scene placement needs"?
Inputs: user query + agent thinking across turns.
Focus:
  1. Did it correctly grasp the theme, mood, and user-named key elements?
  2. Did it produce a reasonable layout plan (zoning, negative space, subject bias)? For large-scene queries, did it grasp the "scale / masses / clusters" implication?
  3. Is the intended asset-style selection clear (e.g., "tranquil ancient style" wants photoreal muted tones; "candy fantasy" wants bright cartoon)?
Rubric:
  0 = no thinking, cannot evaluate; 1 = did not understand at all; 2 = partial with major omissions; 3 = main intent correct but details off;
  4 = essentially complete and accurate, only minor deviations; 5 = perfectly understood all needs and layout intents.

## H3-VR (Visual Reasoning, 0-5): Did the tool_call execution + final scene actually reflect that?
Inputs: user query + full ordered tool_call sequence + final scene element list / screenshots.
Focus (core criterion = does the final scene satisfy YOUR own reading of the user's needs?):
  1. **First read "what does the user want" from the query yourself** (theme, named key elements, mood, scale), then cross-check the final scene: are the elements the user named/implied all present? Is the theme mood in place? Does the scale match (for large-scene queries)?
  2. Is the tool_call chain consistent with the thinking? Are there "thought but did not do" (hallucination) cases?
  3. No tool_call at all / text-only reply / 0 elements -> VR <= 1.
Rubric:
  0 = no valid tool_call; 1 = execution completely inconsistent with understanding, need barely met; 2 = partial with major omissions;
  3 = main needs met, secondary missing; 4 = needs basically met, minor deviations; 5 = perfectly satisfies all needs read from the query.

## H3-Response (distractor recognition, pass/fail)
**You must determine on your own whether the user query contains "impossible / distractor intents".** There are two kinds of clues:
  (A) **Text level:** the query itself violates common sense (rooftop swimming pool with sharks), self-contradicts (extremely quiet bustling night market), or is severely under-specified ("just do something") — judged purely by reading the query.
  (B) **Asset library not covering it:** the user named an element/theme that **the asset library simply lacks**. The evidence lives in the trajectory:
      **if the agent retrieved a given entity multiple times (varying wording) and the recalled candidates are always of the wrong class / severely incompatible, the library does not cover it**
      (e.g., repeatedly retrieving "hover tank / sci-fi tank" but every recall is a non-hovering vehicle). That is an asset-missing distractor.
  Judgment:
  - If the query **does NOT** contain any such distractor (normal request; every named element can be retrieved) -> Response_pass = 1 (evidence = "no_distractor").
  - If the query **does** contain a distractor intent, evaluate the agent's final response:
    * Explicitly points out what cannot be completed / what the library lacks + gives a reasonable substitute or clarification -> Response_pass = 1.
    * Ignores the distractor, jams in a mis-shaped / unmatched element, fabricates, does not mention it at all -> Response_pass = 0.

## H3_pass rule
H3_pass = 1 iff (VU >= 4) AND (VR >= 4) AND (Response_pass == 1); otherwise 0.

## Output format (strict JSON, no extra text)
```json
{
  "H3_VU": {"score": integer 0-5, "evidence": "...(which needs and layout intents were understood/missed)"},
  "H3_VR": {"score": integer 0-5, "evidence": "...(how well the needs read from the query are satisfied; which tool_calls were correct/wrong)"},
  "H3_Response": {"pass": 0 or 1, "evidence": "...(whether the query contains a distractor; if so, whether the agent clarified correctly; if not, write no_distractor)"},
  "H3_pass": 0 or 1,
  "H3_pass_reason": "VU=X, VR=Y, Response=Z, pass condition: VU>=4 AND VR>=4 AND Response==1"
}
```
""".strip()


H3_SCENE_USER_PROMPT = """
## Scene info (from-scratch)
- Theme: {theme}
- User query: {query}
- Terrain hint: {terrain_type}

## Agent multi-turn transcript (per turn: thinking/content + corresponding tool_call)
{agent_turns}

## Agent's final-turn response (user-facing)
{agent_final_response}

## Final scene element list
{final_map_summary}

## Final scene screenshots (5 views: left / right / front / back / top)
{final_image_tags}

## Task
Following the system prompt: first read the user needs from the query yourself, then score H3-VU / H3-VR (0-5),
determine on your own whether the query contains a distractor intent to score H3-Response (pass/fail), and finally output H3_pass (VU>=4 AND VR>=4 AND Response==1).
""".strip()


# ============================================================
# [onestep_scene] H5 — Retrieval-usage plausibility (4 tiers)
# ============================================================

H5_SCENE_SYSTEM_PROMPT = """
You are an expert on retrieval-usage plausibility. The agent calls retrieve_assets to fetch assets and then uses `add` to place them into the scene.
Your task: evaluate whether the agent's **use of retrieval results** is reasonable. **Penalize only obvious, severe, controllable mistakes by the agent** —
poor recall quality from the retrieval service is not the agent's fault; as long as the agent responds appropriately (clarify / rephrase / reasonable substitution / picking an asset in the right coarse category), do not deduct.

## Unit of evaluation
Below you are given per "retrieval intent" pairings: the entity the agent queried + the recalled candidates (with description/color) + the asset actually placed into the scene (used). Assign one tier to each retrieval intent.

## Important priors (internalize; avoid misjudging)
1. **The asset library is overall a cartoon / low-poly art style.** There are almost no strictly photoreal, weathered, or gritty assets.
   Therefore **do NOT use "perfectly photoreal / perfectly on-theme" as the yardstick.** As long as the asset's **coarse class is correct** (query "treasure chest" -> a treasure chest; query "iron pillar" -> a metal pillar) **and does not clash violently with the theme**, count it as fitting (tier1).
2. "Not dark enough", "not weathered enough", "slightly too cartoony" — **minor style deviations are not misuse**; still tier1.
   Only when an asset **clashes violently with the theme or is obviously absurd** (a pink unicorn in a post-apocalyptic battlefield / candy castle) should tier3/tier4 be considered.
3. The agent picking a high-score, correct-class asset from the recall = normal, reasonable behavior = tier1. **Do NOT drop to tier3 just because "another candidate in the recall looked darker to you".**
4. **Judge leniently; when uncertain, default to tier1 or tier2 (pass).** tier3/tier4 are reserved for unmistakably severe misuse.

## Four-tier assignment (each retrieval intent falls into one)
- **tier1 retrieved-right, used-right** (score=1, pass): the recall contains a correct-class, non-clashing asset and the agent used it. (Most retrieval intents belong here.)
- **tier2 retrieved-wrong, clarified** (score=1, pass): the recall's coarse class is wrong or all clash violently (e.g., asking for a treasure chest but only trees come back), but the agent **did not force it** — it clarified in the response, retried with different wording, or simply did not use anything (used=null).
- **tier3 retrieved-right, used-wrong** (score=0, FAIL): the recall **clearly contains** a correct-class asset, but the agent used a **wrong-class or violently clashing** one. (Only for unmistakable misuse; minor style deviations do not count.)
- **tier4 retrieved-wrong, wildly-used / hallucinated** (score=0, FAIL): the recall's coarse class is entirely wrong / all clashing, and the agent **did not clarify and instead forced in** an absurd non-matching asset, OR used a type_id that does not exist in the recall list (fabrication).

## H5_pass rule
H5_pass = 1 iff every retrieval intent is tier1 or tier2 (no tier3 / tier4 anywhere).
worst_tier = the worst tier among all retrieval intents.

## Judgment tips
- used = null (retrieved but did not use any recalled asset): default to "the agent actively abandoned an unfit recall" -> tier2 (pass);
  only if the entity was **explicitly named as required** by the user AND the agent did not use it or backfill it anywhere -> then consider tier4.
- The agent may retrieve the same entity multiple times (varying wording); as long as it finally uses a correct-class asset -> tier1.
- Be lenient toward recall-quality issues themselves; only watch whether the agent "clearly picks wrong / forces in an absurd asset".

## Output format (strict JSON, no extra text)
```json
{
  "intents": [
    {"entity": "retrieval entity name", "tier": 1-4, "used": "name of actually used asset or null", "reason": "why this tier"}
  ],
  "worst_tier": 1-4,
  "H5_pass": 0 or 1,
  "summary": "one-sentence summary"
}
```
""".strip()


H5_SCENE_USER_PROMPT = """
## Scene info (from-scratch)
- Theme: {theme}
- User query: {query}

## Retrieval-intent pairings (entity -> recalled candidates -> asset actually used)
{retrieve_usage_pairs}

## Final scene screenshots (5 views: left / right / front / back / top — reference for style fit)
{final_image_tags}

## Task
Following the system prompt, assign a tier (1-4) to each retrieval intent, and output worst_tier and H5_pass (pass only if no tier3/tier4).
""".strip()


__all__ = [
    # unverified v1 legacy
    "HARD_SYSTEM_PROMPT", "HARD_USER_PROMPT",
    # unverified v2
    "HARD_H12_SYSTEM_PROMPT", "HARD_H12_USER_PROMPT",
    "H3_VURR_SYSTEM_PROMPT", "H3_VURR_USER_PROMPT",
    # shared
    "SOFT_SYSTEM_PROMPT", "SOFT_USER_PROMPT",
    # onestep_scene
    "HARD_H12_SCENE_SYSTEM_PROMPT", "HARD_H12_SCENE_USER_PROMPT",
    "H3_SCENE_SYSTEM_PROMPT", "H3_SCENE_USER_PROMPT",
    "H5_SCENE_SYSTEM_PROMPT", "H5_SCENE_USER_PROMPT",
]

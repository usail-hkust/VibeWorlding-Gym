# VibeWorlding-Gym docs

This directory holds figure assets and supplementary material used by the top-level
`README.md`. Files in here are **not** runtime dependencies — the agent, training
stack, and services all run without them.

## `figures/`

| File | Used in README | Source |
|---|---|---|
| `framework.png` | §1 Introduction | paper Figure 1 (the framework overview) |
| `leaderboard.png` | §3 Sampling and Evaluation | paper Figure (VWE-Bench Verified / Unverified Pass@1) |
| `rl_reward.png` | §5 RL Training | paper Figure (training reward over steps) |

PDFs were converted to PNG with ImageMagick at 200 DPI for inline display on
GitHub. Originals live in `ICLR_2027/overleaf/Figure/` of the paper repo.

## Adding new figures

Keep them in `figures/`, give them descriptive lowercase names, and link with
relative paths so the README renders correctly on both GitHub and PyPI.

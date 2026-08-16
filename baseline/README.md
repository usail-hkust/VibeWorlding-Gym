# Agent-Scaffold Baselines

Reproductions of three published3D-scene agent methods inside the VibeWorld-Gym
sandbox, used as comparison points for VWE-Bench.

Each baseline reuses the *same* sandbox as `main.py` — identical tool schema,
identical asset retrieval, identical Blender rendering, identical output format.
Only the **scaffolding** differs (system prompt, critic/reflection loop, max
turns, stopping rule). Because the per-case outputs match `main.py` byte-for-byte
in structure, `eval.py` scores every method with the same verifier.

| File | Method | Reference | Scaffold |
|---|---|---|---|
| `sceneweaver.py` | SceneWeaver | arXiv:2509.20414 | closed-loop reason → act → reflect, memory `l=1`, `T=10` |
| `sage.py` | SAGE | arXiv:2602.10116 | generators + visual critic + physics critic, self-refine (physics-first) |
| `sceneassistant.py` | SceneAssistant | arXiv:2603.12238 | visual-feedback agent over atomic ops, `T_M=20` |
| `common.py` | — | — | Shared driver: `ScaffoldBot` wraps the real LLM client and injects critic feedback each turn |

## How the scaffolding is injected

`common.py` defines `ScaffoldBot`, which wraps a real client from
`utils/llm.py::MODEL_TYPE_MAP`. Before each turn's observation reaches the
policy, a critic LLM inspects the current 5-view renders and its feedback is
appended to the user message. This reproduces reason-act-reflect,
generator-critic, and visual-feedback loops without forking the sandbox.

Policy and critic share one backbone, selected via `--model_type` / `--model_name`.

## Prerequisites

Both services must be running (see `assets_retrieval/README.md` and
`render_in_blender/README.md`):

```bash
export VIBEWORLD_RETRIEVE_SERVER=http://localhost:8081
export VIBEWORLD_RENDER_SERVER=http://localhost:8080
```

## Running

```bash
# from the repository root
python baseline/sceneweaver.py \
  --base_data_dir data/test \
  --log_dir log/baseline_sceneweaver \
  --model_type gemini --model_name xx \
  --server http://localhost:8080 \
  --retrieve_server http://localhost:8081

python baseline/sage.py           --base_data_dir data/test --log_dir log/baseline_sage           ...
python baseline/sceneassistant.py --base_data_dir data/test --log_dir log/baseline_sceneassistant ...
```

Useful flags (same as `main.py`): `--max_cases`, `--cases`, `--task_setting`,
`--quality`, `--max_turns`, `--debug`.

`--max_turns` defaults to each paper's value, so override it only for ablations.

## Scoring

Baseline runs are evaluated exactly like our own model:

```bash
python eval.py --result_dir log/baseline_sceneweaver \
  --model_type gemini --model_name gemini-3.5-flash
```

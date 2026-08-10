# Asset Retrieval Service

Text-to-asset retrieval service for VibeWorld-Gym. Given a natural-language
entity name (e.g. `探险木屋` / "adventure cabin"), it returns the most
semantically relevant 3D assets from the VWE-Bench asset library, so the agent
can place real, existing assets instead of hallucinating them.

The service is powered by **VibeWorlder-Embedding-4B**, a single-tower dense
retriever fine-tuned from Qwen3-Embedding-4B on our asset-card corpus
(2,622 assets, 2560-dim embeddings).

---

## 1. Prerequisites

Python 3.12(3.10+ works). Install runtime dependencies:

```bash
pip install -r D1_deploy/requirements.txt
```

A CUDA GPU is recommended (first startup encodes the whole asset library,
~1-3 min on GPU). CPU also works, just slower. `flash-attn` is optional — the
code falls back to `sdpa` automatically.

## 2. The embedding model

**VibeWorlder-Embedding-4B ships in this repo** at
`models/VibeWorlder-Embedding-4B/` (~7.6 GB) — nothing to download.

To use a different checkpoint, point `D1_CKPT_DIR` at it. The model is also
published at<https://huggingface.co/collections/usail-hkust/vibeworlder>:

```bash
huggingface-cli download usail-hkust/VibeWorlder-Embedding-4B \
  --local-dir ./models/VibeWorlder-Embedding-4B
```

## 3. Asset data

Already included in `data/` — no download needed:

| File | Rows | Purpose |
|---|---|---|
| `asset_cards.jsonl` | 2,622 | Asset cards; the document side of retrieval |
| `standardized_asset_library_with_caption.csv` | 2,622 | Asset attributes for filtering / return fields |
| `color_shape_detail.csv` | — | Optional color / shape join |
| `view_prefixes.yaml` | — | View → instruction templates (same as training) |

## 4. Start the service

```bash
# from this directory (assets_retrieval/)
PORT=8081 bash deploy.sh
```

Or directly:

```bash
PORT=8081 D1_CKPT_DIR=./models/VibeWorlder-Embedding-4B \
  python3 -m D1_deploy.main
```

First launch precomputes all asset embeddings and caches them under `cache/`,
so subsequent starts are near-instant. Set `D1_FORCE_REENCODE=1` to rebuild
the cache after swapping checkpoints.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8080` | Listen port |
| `D1_HOST` | `0.0.0.0` | Listen address |
| `D1_CKPT_DIR` | `./models/VibeWorlder-Embedding-4B` | Embedding model directory |
| `D1_ASSET_CARDS_JSONL` | `data/asset_cards.jsonl` | Asset cards |
| `D1_ASSET_CSV` | `data/standardized_asset_library_with_caption.csv` | Asset attributes |
| `D1_FORCE_REENCODE` | `0` | Ignore the embedding cache and re-encode |
| `D1_LOG_LEVEL` | `info` | Log level |

## 5. Test it

```bash
curl -s -X POST "http://localhost:8081/recommend/single_slot" \
  -H "Content-Type: application/json" \
  -d '{"entity_name": "探险木屋", "top_k": 3}' | python3 -m json.tool
```

Health / metadata:

```bash
curl -s http://localhost:8081/health | python3 -m json.tool
curl -s http://localhost:8081/info   | python3 -m json.tool
```

## 6. API

All retrieval endpoints are `POST` and return the same envelope.

| Endpoint | Use case |
|---|---|
| `/recommend/single_slot` | One entity name → ranked assets **(main entry point)** |
| `/recommend/entity` | Entity-oriented variant |
| `/recommend/combination` | Multiple slots at once |
| `/recommend/scene` | Scene-level recommendation |
| `/recommend_with_plan` | Recommendation driven by a scene plan |
| `/health`, `/info` | Liveness and loaded-model info (`GET`) |

### `/recommend/single_slot` request

| Field | Type | Required | Notes |
|---|---|---|---|
| `entity_name` | str | ✅ | Entity to look up, e.g. `探险木屋` |
| `top_k` | int | | Default `10`, max `100` |
| `entity_description` | str | | Extra description to sharpen the query |
| `theme` | str | | Scene theme as context |
| `scene_description` | str | | Scene description as context |
| `asset_ids` | list[str] | | Restrict search to these assets |
| `excluded_ids` | list[str] | | Exclude these assets |
| `filters` | dict | | Attribute filters (category, terrain, …) |
| `fields` | list[str] | | Extra attribute fields to return |
| `score_threshold` | float | | Drop results below this cosine score |

Returned assets carry a 5-digit `type_id` that matches the GLB filenames used
by the PCG rendering service, so retrieval output can be placed directly into
a scene.

## 7. Wiring it into the agent

`main.py` and the CLI read the retrieval endpoint from an environment variable:

```bash
export VIBEWORLD_RETRIEVE_SERVER=http://localhost:8081
```

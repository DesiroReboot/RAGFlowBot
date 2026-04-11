# ECBot Configuration Sources

## Public (tracked by git)
- `config/config.md`: variable documentation.
- `.env.example`: shared defaults and key templates.

## Private (not tracked)
- `E:\DATA\ECBot\.env` (recommended)
- You can override path with `ECBOT_DOTENV_PATH`.

## Load Order
1. Private dotenv (`ECBOT_DOTENV_PATH`, default `E:\DATA\ECBot\.env` if exists, else local `.env`)
2. Public template (`.env.example`) as fallback for missing keys
3. JSON config (`config/config.json`)

Because dotenv loading uses non-overwrite semantics, earlier sources have higher priority.

## Team Workflow
- Add new variable keys to `.env.example` and this file first.
- Keep secrets only in private dotenv.
- After pulling latest base, private dotenv continues to override while new keys are backfilled by `.env.example`.

## RAGFlow Retrieval (POC)
- `ECBOT_RAG_PROVIDER`: `legacy|ragflow` (default `legacy`)
- `ECBOT_RAGFLOW_BASE_URL`: RAGFlow API host
- `ECBOT_RAGFLOW_API_KEY`: RAGFlow token
- `ECBOT_RAGFLOW_DATASET_MAP`: JSON map, e.g. `{"default":"ds_123","tenant1:botA":"ds_456"}`
- `ECBOT_RAGFLOW_TIMEOUT_MS`: request timeout in milliseconds
- `ECBOT_RAGFLOW_TOP_K`: retrieval top-k
- `ECBOT_RAGFLOW_MIN_SCORE`: score filter threshold
- `ECBOT_RAGFLOW_FALLBACK_TO_LEGACY`: fallback to legacy retriever when RAGFlow errors

## Model Sync
- `ECBOT_MODEL`: shared model name for both `embedding.model` and `generation.model`.
- Priority:
  1. `ECBOT_EMBEDDING_MODEL` / `ECBOT_GENERATION_MODEL`
  2. `ECBOT_MODEL`
  3. `config/config.json` values

## Knowledge Base Dir
- Default `knowledge_base.source_dir`: `E:\知识库\RailKB`
- Override by:
  1. `ECBOT_KB_SOURCE_DIR`
  2. `config/config.json` -> `knowledge_base.source_dir`

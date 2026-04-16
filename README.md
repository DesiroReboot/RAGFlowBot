# RAGFlowBot

RAGFlowBot 是一个面向飞书的问答机器人，当前网关已重构为 **FastAPI (ASGI)** 实现。

## 项目沿革（存证）

- 本仓库由 `ECBot` 历史演进而来，采用保留完整 Git 提交历史的方式拆分维护。
- 同根基线提交（split base）：`f2cec3950efb017c5507e7d4960fda4a47b77ade`
- 建议在 `ECBot` 仓库保留归档标签：`ecbot-archive-2026-04-11`
- 可审计性说明：若 `ECBot` 与 `RAGFlowBot` 同时保留该提交 SHA，可证明两个仓库来源于同一历史根。

## 架构概览

当前系统采用三层架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    Gateway Layer                              │
│                (FastAPI / src/fastapi_gateway)               │
│  - HTTP API: /health, /gateway/*, /webhook/feishu           │
│  - Event handling & response formatting                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Core Layer                                 │
│                  (src/core)                                  │
│  - ReActAgent: Query orchestration                           │
│  - SearchOrchestrator: Hybrid retrieval coordination          │
│  - Web routing & Lite pipeline                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Base Layer                       │
│                    (src/KB)                                  │
│  - KBStatusService: Index status & staleness detection       │
│  - KnowledgeBaseBuilder: Index construction                  │
│  - ManifestStore: Index metadata management                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    RAG Processing Layer                       │
│                    (src/RAG)                                 │
│  - Indexing: FTS + Vector embeddings                         │
│  - Retrieval: Hybrid search (Fusion + Grading)               │
│  - Classification & Quality filtering                        │
└─────────────────────────────────────────────────────────────┘
```

**架构说明：**
- **Gateway Layer**: FastAPI 服务，处理 HTTP 请求和飞书事件
- **Core Layer**: 查询编排和检索逻辑
- **Knowledge Base Layer**: 知识库索引管理和状态监控（新增 `src/KB` 模块）
- **RAG Processing Layer**: 底层检索和索引能力

## 当前网关状态

- 网关实现：`src/fastapi_gateway`
- 启动入口：`src/main.py`（内部调用 FastAPI runtime）
- 已移除旧目录：`src/gateway`
- 兼容保留配置：`gateway.feishu.*`

## HTTP 接口

默认接口如下（可通过配置调整 webhook 路径和端口）：

**健康检查：**
- `GET /health` - 基础健康检查
- `GET /gateway/startup-check` - 启动状态检查
- `GET /gateway/self-check` - 完整自检

**知识库状态（新增）：**
- `GET /gateway/kb/status` - KB 索引状态与过期检测
  - 返回：state（no_index/empty/partial/ready/stale/failed）
  - 返回：indexed_files、indexed_chunks、source_file_count
  - 返回：ready_for_query 标志

**检索与诊断：**
- `POST /gateway/fullchain-visualize` - 全链路可视化
- `POST /webhook/feishu` - 飞书事件回调

## 知识库管理 CLI

### 1. 查看 KB 索引状态

```powershell
# 基础状态查看
python scripts/kb_status.py

# JSON 格式输出
python scripts/kb_status.py --json

# 自定义配置文件
python scripts/kb_status.py --config path/to/config.json
```

**状态说明：**
- `no_index` - 从未构建
- `empty` - 已扫描但无文件
- `partial` - 部分文件索引失败
- `ready` - 索引就绪
- `stale` - 索引已过期（源文件有变化）
- `failed` - 构建失败

**退出码：**
- 0: ready
- 1: no_index
- 2: failed
- 3: stale
- 4: partial
- 100: error

### 2. 初始化/重建 KB 索引

```powershell
# 初始化索引（首次构建）
python scripts/kb_init.py

# 强制重建索引
python scripts/kb_init.py --force-reindex

# 指定源目录
python scripts/kb_init.py --source-dir "E:\知识库\RailKB"

# 指定配置文件
python scripts/kb_init.py --config path/to/config.json
```

### 3. 增量同步 KB 索引

```powershell
# 增量同步（仅索引变更文件）
python scripts/kb_build_index.py

# 强制重建
python scripts/kb_build_index.py --force-reindex

# 指定源目录
python scripts/kb_build_index.py --source-dir "E:\知识库\RailKB"
```

### 4. 检查索引就绪状态

```powershell
# 详细就绪检查
python scripts/kb_check_readiness.py

# JSON 格式输出
python scripts/kb_check_readiness.py --json
```

### 5. 提升指定构建版本

```powershell
# 提升当前配置版本为活跃
python scripts/kb_promote_manifest.py

# 提升指定版本
python scripts/kb_promote_manifest.py --build-version "2025-01-15"
```

## 配置示例

配置文件默认读取 `config/config.json`（若不存在会回退到 `config.json`）。
程序启动时仅加载项目内 `.env`，然后用 `.env.example` 补齐缺失变量。

```json
{
  "gateway": {
    "feishu": {
      "enabled": true,
      "receive_mode": "webhook",
      "openapi_base_url": "https://open.feishu.cn/open-apis",
      "app_id": "YOUR_APP_ID",
      "app_secret": "YOUR_APP_SECRET",
      "verification_token": "YOUR_VERIFICATION_TOKEN",
      "encrypt_key": "YOUR_ENCRYPT_KEY",
      "webhook_port": 8000,
      "webhook_path": "/webhook/feishu"
    }
  },
  "knowledge_base": {
    "source_dir": "E:\\知识库\\RailKB",
    "auto_init_on_startup": true,
    "init_blocking": false,
    "init_fail_open": true,
    "build_version": "2025-01-15"
  },
  "database": {
    "db_path": "DB/ec_bot.db"
  }
}
```

说明：
- `receive_mode=webhook` 时使用 HTTP 回调。
- `receive_mode=long_connection` 时使用飞书 SDK 长连接，不需要配置订阅 URL。
- 配置解析规则：`config/config.json` 优先；当配置值为空、占位符（如 `YOUR_*`）或缺失时，才会回退到环境变量（私有 `E:\\DATA\\ECBot\\.env` / 项目 `.env` / `.env.example`）。

## .env 配置（推荐）

```powershell
copy .env.example .env
```

把真实密钥写入 `.env`，并保持 `config/config.json` 使用占位符（如 `YOUR_*`），运行时会自动从 `.env` 回退补全。

模型配置可使用共享变量避免重复修改：

```text
ECBOT_MODEL=qwen-plus
```

优先级：
- `ECBOT_EMBEDDING_MODEL` / `ECBOT_GENERATION_MODEL`
- `ECBOT_MODEL`
- `config/config.json` 中的 `embedding.model` / `generation.model`

知识库默认目录为 `E:\知识库\RailKB`，可在 `config/config.json` 的 `knowledge_base.source_dir` 或环境变量 `ECBOT_KB_SOURCE_DIR` 覆盖。

## RAGFlow 模式（POC）

默认检索 provider 是 `legacy`（本地索引）。要切换到 `ragflow`：

```text
ECBOT_RAG_PROVIDER=ragflow
ECBOT_RAGFLOW_BASE_URL=https://<ragflow-host>
ECBOT_RAGFLOW_API_KEY=<ragflow-key>
ECBOT_RAGFLOW_DATASET_MAP={"default":"ds_123"}
ECBOT_RAGFLOW_TIMEOUT_MS=2500
ECBOT_RAGFLOW_TOP_K=5
ECBOT_RAGFLOW_MIN_SCORE=0.1
ECBOT_RAGFLOW_FALLBACK_TO_LEGACY=true
```

说明：
- `ragflow` 模式下跳过本地 `index_manifest` 阻断 gate。
- 若开启 `ECBOT_RAGFLOW_FALLBACK_TO_LEGACY=true`，RAGFlow 请求失败会自动回退到 `legacy` 检索。
- 可通过 `GET /gateway/self-check` 查看 `retrieval.provider` 与 `retrieval.ragflow_config` 诊断项。

## 启动方式

### 1) 推荐：项目统一入口

```powershell
py -m src.main
```

### 0) Docker 运行（迁移后推荐）

首次构建并启动：

```powershell
docker compose up -d --build
```

查看日志：

```powershell
docker compose logs -f ecbot
```

停止：

```powershell
docker compose down
```

说明：
- 已提供 `Dockerfile`、`docker-compose.yml`、`.dockerignore`、`requirements.txt`。
- 默认挂载目录：`./config`、`./DB`、`./logs`、`./Eval`、`./kb`。
- 容器默认读取 `.env`，并将知识库目录固定为容器内 `/app/kb`（由 `ECBOT_KB_SOURCE_DIR=/app/kb` 覆盖）。
- 当前镜像以"最小可运行"为目标；若需容器内 OCR，请在镜像中额外安装 `tesseract-ocr` 与对应语言包。
- 已内置 Docker healthcheck：
  - `receive_mode=long_connection`：检查最近长连接状态文件（默认路径 `/tmp/ecbot_longconn_health.json`）。
  - `receive_mode=webhook`：检查 `http://127.0.0.1:8000/health`。

查看健康状态：

```powershell
docker inspect ecbot --format "{{json .State.Health}}"
```

若你使用 `webhook` 模式，确保 `.env` 或 `config/config.json` 里设置：

```text
ECBOT_FEISHU_RECEIVE_MODE=webhook
```

若你使用 `long_connection` 模式，保持：

```text
ECBOT_FEISHU_RECEIVE_MODE=long_connection
```

### 2) 直接使用 uvicorn（仅 webhook 模式）

```powershell
py -m uvicorn src.fastapi_gateway.app:create_app --factory --host 0.0.0.0 --port 8000
```

如果你希望和配置文件中的端口一致，把 `--port` 改成 `gateway.feishu.webhook_port` 对应值。`receive_mode=long_connection` 时不要直接用 uvicorn，应该使用 `py -m src.main`。

### 3) 长连接模式运行

```powershell
setx ECBOT_FEISHU_RECEIVE_MODE long_connection
py -m src.main
```

长连接模式不依赖 `webhook_path/webhook_port`，但仍建议保留便于切回 webhook 模式。

## 本地联调

```powershell
# 健康检查
curl http://127.0.0.1:8000/health

# 启动检查
curl http://127.0.0.1:8000/gateway/startup-check

# 完整自检
curl http://127.0.0.1:8000/gateway/self-check

# KB 状态检查（新增）
curl http://127.0.0.1:8000/gateway/kb/status

# 全链路可视化
curl -X POST http://127.0.0.1:8000/gateway/fullchain-visualize -H "Content-Type: application/json" -d "{\"query\":\"测试问题\"}"
```

若使用飞书事件订阅并通过隧道暴露公网地址，回调 URL 形如：

```text
https://<your-public-domain>/webhook/feishu
```

## 测试

网关测试文件：

- `tests/test_fastapi_gateway_runtime.py`
- `tests/test_fastapi_gateway_handler.py`

运行示例：

```powershell
python -m pytest -q tests/test_fastapi_gateway_runtime.py tests/test_fastapi_gateway_handler.py
```

KB 状态测试：

```powershell
python -m pytest -q tests/test_kbase_startup_bootstrap.py tests/test_manifest_store.py
```

## Golden Set 固定 Pipeline

固定入口（每次同一流程：跑 golden set、写 trace/report/html、登记运行记录）：

```powershell
python scripts/run_golden_pipeline.py
```

按分片 set 执行（示例：`golden-set-0` 到 `golden-set-4`）：

```powershell
python scripts/run_golden_pipeline.py --set-ids 0,1,2,3,4
```

按自定义数据集列表执行：

```powershell
python scripts/run_golden_pipeline.py --datasets Eval/golden-set-0.json,Eval/golden-set-2.json
```

仅查看将执行哪些 set（不实际运行）：

```powershell
python scripts/run_golden_pipeline.py --set-ids 0,1,2,3,4 --dry-run
```

每个 set 执行后都会生成并打印：
- report 路径
- trace index 路径
- html index 路径
- diff index 路径
- 阶段性摘要（仅控制台输出，不写入文件）

运行记录会写入：`Eval/pipeline/golden-set.pipeline.json`。

## 常见问题

### KB 索引状态异常

```powershell
# 查看当前状态
python scripts/kb_status.py

# 如果状态为 stale，执行增量同步
python scripts/kb_build_index.py

# 如果状态为 failed 或 no_index，执行完整初始化
python scripts/kb_init.py --force-reindex
```

### 配置文件位置

- 主配置：`config/config.json`
- 环境变量：`.env`（从 `.env.example` 复制）
- 数据库：`DB/ec_bot.db`
- 知识库源：默认 `E:\知识库\RailKB`

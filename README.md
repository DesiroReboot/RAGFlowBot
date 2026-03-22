# ECBot

ECBot 是一个面向飞书的问答机器人，当前网关已重构为 **FastAPI (ASGI)** 实现。

## 当前网关状态

- 网关实现：`src/fastapi_gateway`
- 启动入口：`src/main.py`（内部调用 FastAPI runtime）
- 已移除旧目录：`src/gateway`
- 兼容保留配置：`gateway.feishu.*`

## HTTP 接口

默认接口如下（可通过配置调整 webhook 路径和端口）：

- `GET /health`
- `GET /gateway/startup-check`
- `GET /gateway/self-check`
- `POST /gateway/fullchain-visualize`
- `POST /webhook/feishu`

## 配置示例

配置文件默认读取 `config/config.json`（若不存在会回退到 `config.json`）。
程序启动时会自动加载项目根目录 `.env`（可用 `ECBOT_DOTENV_PATH` 指定其他路径）。

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
  }
}
```

说明：
- `receive_mode=webhook` 时使用 HTTP 回调。
- `receive_mode=long_connection` 时使用飞书 SDK 长连接，不需要配置订阅 URL。
- 配置优先级：系统环境变量 > `.env` > `config/config.json`。

## .env 配置（推荐）

```powershell
copy .env.example .env
```

把真实密钥写入 `.env`，并保持 `config/config.json` 使用占位符。

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
- 当前镜像以“最小可运行”为目标；若需容器内 OCR，请在镜像中额外安装 `tesseract-ocr` 与对应语言包。
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
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/gateway/startup-check
curl http://127.0.0.1:8000/gateway/self-check
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


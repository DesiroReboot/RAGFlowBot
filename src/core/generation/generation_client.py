from __future__ import annotations

import json
from time import sleep
from urllib import parse, request

from src.config import GenerationConfig


class GenerationClient:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.base_url = str(config.base_url).rstrip("/")
        self.api_key = str(config.api_key or "").strip()
        self.model = str(config.model or "").strip()
        self.timeout = max(1, int(config.timeout))
        self.max_retries = max(0, int(config.max_retries))
        self.temperature = float(config.temperature)

    @property
    def available(self) -> bool:
        if not self.base_url or not self.model:
            return False
        if not self.api_key or self.api_key.upper().startswith("YOUR_"):
            return False
        return True

    def rewrite(
        self,
        *,
        query: str,
        template_answer: str,
        answer_class: str = "open",  # 新参数，替换answer_mode
        answer_mode: str | None = None,  # 保留以兼容旧调用
        key_points: list[str] | None = None,
        steps: list[str],
        evidence: list[str],
        citation_sources: list[str],
        paragraph_output: bool = True,
    ) -> str:
        if not self.available:
            raise RuntimeError("generation client unavailable")

        # 向后兼容：answer_mode映射到answer_class
        if answer_mode and answer_class == "open":
            # 映射逻辑：fact_qa -> qa, procedure/mixed -> open
            answer_class = "qa" if answer_mode == "fact_qa" else "open"

        # 根据answer_class选择system prompt
        from src.core.classification.prompt_templates import get_system_prompt

        system_prompt = get_system_prompt(answer_class)

        # 移除原有的format_instruction逻辑（已包含在prompt模板中）
        evidence_block = (
            "\n".join(f"- {line}" for line in evidence) if evidence else "- 无"
        )
        key_points_block = (
            "\n".join(f"- {line}" for line in (key_points or []) if str(line).strip())
            if key_points
            else "- 无"
        )
        steps_block = (
            "\n".join(f"{idx}. {line}" for idx, line in enumerate(steps, start=1))
            if steps
            else "- 无"
        )
        citations_block = (
            "\n".join(f"- {source}" for source in citation_sources)
            if citation_sources
            else "- 无"
        )

        user_prompt = (
            f"用户问题：{query}\n\n"
            f"回答类别：{answer_class}\n\n"
            f"模板答案：\n{template_answer}\n\n"
            f"可用要点：\n{key_points_block}\n\n"
            f"可用步骤：\n{steps_block}\n\n"
            f"可用证据：\n{evidence_block}\n\n"
            f"可用来源：\n{citations_block}\n\n"
            "请输出最终答案。要求：表达自然，关系映射完整，来源可追溯，不要添加任何未给出的新事实。"
        )

        url = self._safe_http_url(f"{self.base_url}/chat/completions")
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")
        req = request.Request(
            url=url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with request.urlopen(
                    req, timeout=self.timeout
                ) as response:  # nosec B310
                    body = response.read().decode("utf-8", errors="ignore")
                data = json.loads(body)
                content = self._extract_content(data)
                if not content:
                    raise RuntimeError("empty generation response")
                return content
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    sleep(min(1.5 * (attempt + 1), 4.0))
                    continue
                break
        raise RuntimeError(f"generation rewrite failed: {last_error}")

    def _safe_http_url(self, raw_url: str) -> str:
        parsed = parse.urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"unsupported url scheme: {parsed.scheme!r}")
        if not parsed.netloc:
            raise ValueError("missing url host")
        return raw_url

    def _extract_content(self, payload: dict) -> str:
        choices = payload.get("choices", [])
        if not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message", {}) if isinstance(first, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    text = str(block.get("text", "")).strip()
                    if text:
                        text_parts.append(text)
            return "\n".join(text_parts).strip()
        return str(content).strip()

from __future__ import annotations

import json
from pathlib import Path

from src.config import Config
from src.core.search.domain_filter import DomainFilter
from src.core.search.planner import RulePlanner


_TMP_CONFIG_DIR = Path("pytest_temp_dir_domain_filter")


def _write_config_file(name: str, payload: dict[str, object]) -> Path:
    _TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = _TMP_CONFIG_DIR / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_domain_filter_allows_trade_query() -> None:
    domain_filter = DomainFilter(threshold=0.45)

    result = domain_filter.check("亚马逊Listing如何优化关键词和PPC投放？")

    assert result.allow_rag is True
    assert result.score >= 0.45
    assert "listing" in result.positive_hits


def test_domain_filter_blocks_non_domain_query() -> None:
    domain_filter = DomainFilter(threshold=0.45)

    result = domain_filter.check("帮我写一个Python脚本，求两数之和")

    assert result.allow_rag is False
    assert result.reason in {"score_below_threshold", "negative_intent_without_domain_signal"}
    assert "python" in result.negative_hits


def test_domain_filter_short_query_fail_open() -> None:
    domain_filter = DomainFilter(threshold=0.45)

    result = domain_filter.check("库存")

    assert result.allow_rag is True
    assert result.reason == "short_query_fail_open"


def test_domain_filter_allows_mixed_language_query() -> None:
    domain_filter = DomainFilter(threshold=0.45)

    result = domain_filter.check("How to reduce Amazon tariff and customs delay for FBA shipment?")

    assert result.allow_rag is True
    assert result.score >= 0.45
    assert "amazon" in result.positive_hits
    assert "tariff" in result.positive_hits
    assert "customs" in result.positive_hits


def test_planner_blocks_rag_when_out_of_scope() -> None:
    planner = RulePlanner(domain_filter_enabled=True, domain_filter_threshold=0.45)

    output = planner.plan("今天NBA比赛谁会赢？")

    assert output.allow_rag is False
    assert output.need_web_search is False
    assert output.source_route == "kb_only"
    assert output.filter_reason in {"score_below_threshold", "negative_intent_without_domain_signal"}


def test_planner_allows_when_domain_filter_disabled() -> None:
    planner = RulePlanner(domain_filter_enabled=False, domain_filter_threshold=0.99)

    output = planner.plan("写一个Python冒泡排序")

    assert output.allow_rag is True
    assert output.filter_reason == "domain_filter_disabled"


def test_planner_fail_open_on_filter_error() -> None:
    class _BrokenDomainFilter:
        threshold = 0.45

        def check(self, query: str) -> object:  # noqa: ARG002
            raise RuntimeError("boom")

    planner = RulePlanner(
        domain_filter_enabled=True,
        domain_filter_threshold=0.45,
        domain_filter_fail_open=True,
        domain_filter=_BrokenDomainFilter(),  # type: ignore[arg-type]
    )

    output = planner.plan("无关问题也先放行")

    assert output.allow_rag is True
    assert output.filter_reason == "domain_filter_error_fail_open"


def test_config_reads_domain_filter_from_json() -> None:
    config_path = _write_config_file(
        "domain_filter_json_config.json",
        {
            "search": {
                "domain_filter_enabled": False,
                "domain_filter_threshold": 0.72,
                "domain_filter_fail_open": False,
            }
        },
    )

    config = Config(config_path=str(config_path))

    assert config.search.domain_filter_enabled is False
    assert config.search.domain_filter_threshold == 0.72
    assert config.search.domain_filter_fail_open is False


def test_config_env_overrides_domain_filter(monkeypatch) -> None:
    config_path = _write_config_file(
        "domain_filter_env_config.json",
        {
            "search": {
                "domain_filter_enabled": False,
                "domain_filter_threshold": 0.51,
                "domain_filter_fail_open": False,
            }
        },
    )
    monkeypatch.setenv("ECBOT_DOMAIN_FILTER_ENABLED", "true")
    monkeypatch.setenv("ECBOT_DOMAIN_FILTER_THRESHOLD", "0.83")
    monkeypatch.setenv("ECBOT_DOMAIN_FILTER_FAIL_OPEN", "1")

    config = Config(config_path=str(config_path))

    assert config.search.domain_filter_enabled is True
    assert config.search.domain_filter_threshold == 0.83
    assert config.search.domain_filter_fail_open is True

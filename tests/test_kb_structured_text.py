from __future__ import annotations

from pathlib import Path

from src.RAG.config.kbase_config import KBaseConfig
from src.RAG.preprocessing.parser import DocumentParser


def test_kbase_config_default_extensions_include_structured_text() -> None:
    cfg = KBaseConfig()
    assert ".json" in cfg.supported_extensions
    assert ".xml" in cfg.supported_extensions


def test_document_parser_parse_json_as_structured_text(tmp_path: Path) -> None:
    parser = DocumentParser(KBaseConfig())
    target = tmp_path / "sample.json"
    target.write_text('{"user":{"name":"alice"},"items":[{"id":1},{"id":2}]}', encoding="utf-8")

    content, metadata = parser.parse(target)

    assert metadata["type"] == "json"
    assert metadata["parse_method"] == "json_flatten"
    assert "user.name: alice" in content
    assert "items[0].id: 1" in content


def test_document_parser_parse_xml_as_structured_text(tmp_path: Path) -> None:
    parser = DocumentParser(KBaseConfig())
    target = tmp_path / "sample.xml"
    target.write_text('<root><order id="A1"><item>chair</item></order></root>', encoding="utf-8")

    content, metadata = parser.parse(target)

    assert metadata["type"] == "xml"
    assert metadata["parse_method"] == "xml_flatten"
    assert "root/order@id: A1" in content
    assert "root/order/item: chair" in content

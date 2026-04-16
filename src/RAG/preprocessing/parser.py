from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Any

from src.RAG.config.kbase_config import KBaseConfig

PdfReader: Any = None
try:
    from pypdf import PdfReader as _ImportedPdfReader  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency.
    pass
else:
    PdfReader = _ImportedPdfReader

fitz: Any = None
try:
    import fitz as _imported_fitz  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency.
    pass
else:
    fitz = _imported_fitz

Image: Any = None
try:
    from PIL import Image as _ImportedPILImage
except Exception:  # pragma: no cover - optional dependency.
    pass
else:
    Image = _ImportedPILImage

pytesseract: Any = None
try:
    import pytesseract as _imported_pytesseract  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency.
    pass
else:
    pytesseract = _imported_pytesseract


class DocumentParser:
    CODE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".cpp",
        ".c",
        ".cs",
        ".php",
    }

    _NOISE_MARKERS = ("stream", "xref", "endobj", "/filter", "/length", "obj", "flatedecode")

    def __init__(self, config: KBaseConfig):
        self.config = config

    def parse(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self._parse_pdf(file_path)
        if ext == ".json":
            return self._parse_json(file_path)
        if ext == ".xml":
            return self._parse_xml(file_path)
        if ext in self.CODE_EXTENSIONS:
            content = self._read_text(file_path)
            return content, {"type": "code", "language": ext.lstrip(".")}

        content = self._read_text(file_path)
        return content, {"type": "text", "language": "plain"}

    def extract_text_chunks(self, content: str, chunk_size: int, overlap: int) -> list[str]:
        if not content:
            return []
        step = max(1, chunk_size - overlap)
        return [content[i : i + chunk_size] for i in range(0, len(content), step)]

    def _read_text(self, file_path: Path) -> str:
        if not file_path.exists():
            return ""

        for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
            try:
                return file_path.read_text(encoding=encoding, errors="strict")
            except UnicodeDecodeError:
                continue
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _parse_json(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        raw = self._read_text(file_path)
        metadata: dict[str, Any] = {"type": "json", "language": "json"}
        if not raw.strip():
            metadata["parse_method"] = "json_empty"
            return "", metadata
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            metadata["parse_method"] = "json_raw_fallback"
            return raw, metadata

        flattened = self._flatten_json(payload)
        metadata["parse_method"] = "json_flatten"
        return flattened or raw, metadata

    def _parse_xml(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        raw = self._read_text(file_path)
        metadata: dict[str, Any] = {"type": "xml", "language": "xml"}
        if not raw.strip():
            metadata["parse_method"] = "xml_empty"
            return "", metadata
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            metadata["parse_method"] = "xml_raw_fallback"
            return raw, metadata

        flattened = self._flatten_xml(root)
        metadata["parse_method"] = "xml_flatten"
        return flattened or raw, metadata

    def _flatten_json(self, payload: Any) -> str:
        lines: list[str] = []

        def walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                if not node and path:
                    lines.append(f"{path}: {{}}")
                    return
                for key, value in node.items():
                    key_text = str(key).strip()
                    if not key_text:
                        continue
                    next_path = f"{path}.{key_text}" if path else key_text
                    walk(value, next_path)
                return
            if isinstance(node, list):
                if not node:
                    lines.append(f"{path}: []")
                    return
                for index, item in enumerate(node):
                    next_path = f"{path}[{index}]" if path else f"[{index}]"
                    walk(item, next_path)
                return

            value = self._scalar_to_text(node)
            if path:
                lines.append(f"{path}: {value}")
            else:
                lines.append(value)

        walk(payload, "")
        return "\n".join(line for line in lines if line.strip()).strip()

    def _flatten_xml(self, root: ET.Element) -> str:
        lines: list[str] = []

        def walk(node: ET.Element, path: str) -> None:
            tag = self._strip_xml_namespace(node.tag)
            next_path = f"{path}/{tag}" if path else tag

            for attr_name, attr_value in node.attrib.items():
                attr_key = self._strip_xml_namespace(attr_name)
                lines.append(f"{next_path}@{attr_key}: {attr_value}")

            text = (node.text or "").strip()
            if text:
                lines.append(f"{next_path}: {text}")

            for child in list(node):
                walk(child, next_path)

        walk(root, "")
        return "\n".join(line for line in lines if line.strip()).strip()

    def _strip_xml_namespace(self, name: str) -> str:
        if "}" in name:
            return name.rsplit("}", 1)[-1]
        return name

    def _scalar_to_text(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value
        return str(value)

    def _parse_pdf(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        if not file_path.exists():
            return "", {
                "type": "pdf",
                "language": "pdf",
                "parse_method": "missing",
                "readability_score": 0.0,
                "noise_ratio": 1.0,
            }

        pypdf_pages = self._extract_pdf_pages(file_path)
        pypdf_text = "\n\n".join(page for page in pypdf_pages if page.strip())
        cleaned, sanitize_stats = self._sanitize_pdf_text(
            pypdf_text,
            page_texts=pypdf_pages if pypdf_pages else None,
        )
        readability = self._readability_score(cleaned)
        noise_ratio = self._noise_ratio(cleaned)
        parse_method = "pypdf" if cleaned.strip() else "pypdf_empty"

        if self._should_try_ocr(cleaned, readability):
            ocr_text, ocr_method = self._extract_with_ocr(file_path)
            if ocr_text.strip():
                ocr_cleaned, ocr_stats = self._sanitize_pdf_text(ocr_text)
                ocr_readability = self._readability_score(ocr_cleaned)
                if ocr_readability >= readability or not cleaned.strip():
                    cleaned = ocr_cleaned
                    sanitize_stats = ocr_stats
                    readability = ocr_readability
                    noise_ratio = self._noise_ratio(cleaned)
                    parse_method = ocr_method
                else:
                    parse_method = f"pypdf_preferred_over_{ocr_method}"
            elif parse_method == "pypdf_empty" and not self._looks_like_pdf_binary(file_path):
                # A mislabeled text file should still be recoverable.
                cleaned, sanitize_stats = self._sanitize_pdf_text(self._read_text(file_path))
                readability = self._readability_score(cleaned)
                noise_ratio = self._noise_ratio(cleaned)
                parse_method = "text_fallback_non_pdf"

        metadata: dict[str, Any] = {
            "type": "pdf",
            "language": "pdf",
            "parse_method": parse_method,
            "readability_score": round(readability, 6),
            "noise_ratio": round(noise_ratio, 6),
        }
        metadata.update(sanitize_stats)
        return cleaned, metadata

    def _extract_pdf_pages(self, file_path: Path) -> list[str]:
        if PdfReader is None:
            return []
        try:
            reader = PdfReader(str(file_path))
        except Exception:
            return []
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append((page.extract_text() or "").replace("\x00", ""))
            except Exception:
                pages.append("")
        return pages

    def _extract_with_ocr(self, file_path: Path) -> tuple[str, str]:
        if not self.config.ocr_enabled:
            return "", "ocr_disabled"
        if fitz is None or pytesseract is None or Image is None:
            return "", "ocr_unavailable"

        scale = max(1.0, float(self.config.ocr_dpi_scale))
        lang = str(self.config.ocr_language or "chi_sim+eng").strip()
        texts: list[str] = []
        try:
            with fitz.open(str(file_path)) as doc:
                for page in doc:
                    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    text = pytesseract.image_to_string(image, lang=lang)
                    if text.strip():
                        texts.append(text)
        except Exception:
            return "", "ocr_failed"
        return "\n\n".join(texts), "ocr_pymupdf_tesseract"

    def _sanitize_pdf_text(
        self,
        text: str,
        *,
        page_texts: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if not text and not page_texts:
            return "", {
                "raw_lines": 0,
                "kept_lines": 0,
                "removed_lines": 0,
                "removed_header_footer_lines": 0,
                "removed_noise_lines": 0,
                "removed_catalog_lines": 0,
                "removed_symbol_lines": 0,
                "removed_short_lines": 0,
            }

        pages = self._prepare_pages(text=text, page_texts=page_texts)
        repeated_signatures = self._detect_repeated_header_footer(pages)

        kept_lines: list[str] = []
        raw_lines = 0
        removed_header_footer = 0
        removed_noise = 0
        removed_catalog = 0
        removed_symbol = 0
        removed_short = 0

        for lines in pages:
            for line in lines:
                raw_lines += 1
                signature = self._line_signature(line)
                lowered = line.lower()
                if signature in repeated_signatures:
                    removed_header_footer += 1
                    continue
                if self._is_directory_or_page_line(line):
                    removed_catalog += 1
                    continue
                if self._is_pdf_noise_line(lowered):
                    removed_noise += 1
                    continue
                if self._is_symbol_noise_line(line):
                    removed_symbol += 1
                    continue
                if self._is_short_uninformative_line(line):
                    removed_short += 1
                    continue
                kept_lines.append(line)

        cleaned = "\n".join(kept_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        removed_total = (
            removed_header_footer
            + removed_noise
            + removed_catalog
            + removed_symbol
            + removed_short
        )
        return cleaned, {
            "raw_lines": raw_lines,
            "kept_lines": len(kept_lines),
            "removed_lines": removed_total,
            "removed_header_footer_lines": removed_header_footer,
            "removed_noise_lines": removed_noise,
            "removed_catalog_lines": removed_catalog,
            "removed_symbol_lines": removed_symbol,
            "removed_short_lines": removed_short,
        }

    def _prepare_pages(self, *, text: str, page_texts: list[str] | None) -> list[list[str]]:
        raw_pages = page_texts or [text]
        pages: list[list[str]] = []
        for raw_page in raw_pages:
            normalized = [self._normalize_line(line) for line in raw_page.replace("\x00", "").splitlines()]
            normalized = [line for line in normalized if line]
            if not normalized:
                continue
            pages.append(self._merge_hyphenated_lines(normalized))
        return pages

    def _merge_hyphenated_lines(self, lines: list[str]) -> list[str]:
        merged: list[str] = []
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if (
                line.endswith("-")
                and idx + 1 < len(lines)
                and re.match(r"^[A-Za-z][A-Za-z0-9]*", lines[idx + 1])
            ):
                merged.append(f"{line[:-1]}{lines[idx + 1]}")
                idx += 2
                continue
            merged.append(line)
            idx += 1
        return merged

    def _normalize_line(self, raw_line: str) -> str:
        return re.sub(r"\s+", " ", raw_line).strip()

    def _detect_repeated_header_footer(self, pages: list[list[str]]) -> set[str]:
        if len(pages) < 2:
            return set()
        counter: Counter[str] = Counter()
        for lines in pages:
            candidates = lines[:2] + lines[-2:]
            seen_on_page: set[str] = set()
            for line in candidates:
                signature = self._line_signature(line)
                if not signature or signature in seen_on_page:
                    continue
                if len(line) > 120:
                    continue
                seen_on_page.add(signature)
                counter[signature] += 1
        threshold = max(2, int(math.ceil(len(pages) * 0.4)))
        return {signature for signature, count in counter.items() if count >= threshold}

    def _line_signature(self, line: str) -> str:
        lowered = line.lower()
        lowered = re.sub(r"\d+", "#", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def _is_directory_or_page_line(self, line: str) -> bool:
        lowered = line.lower().strip()
        if not lowered:
            return True
        if re.fullmatch(r"\d{1,4}", lowered):
            return True
        if re.fullmatch(r"(?:第\s*)?\d+\s*(?:页|/)\s*\d*", lowered):
            return True
        if re.fullmatch(r"(?:page\s*)?\d+\s*(?:of|/)\s*\d+", lowered):
            return True
        if re.fullmatch(r"(contents?|目录)", lowered):
            return True
        if "all rights reserved" in lowered or "版权所有" in lowered or "copyright" in lowered:
            return True
        if re.fullmatch(r"[.\-·•…\s]{3,}\d{1,4}", lowered):
            return True
        if re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9\s]{2,40}[.\-·•…]{2,}\s*\d{1,4}", line):
            return True
        return False

    def _is_pdf_noise_line(self, lowered_line: str) -> bool:
        marker_hits = sum(1 for marker in self._NOISE_MARKERS if marker in lowered_line)
        symbol_ratio = len(re.findall(r"[<>{}\\/\[\]]", lowered_line)) / max(len(lowered_line), 1)
        return marker_hits >= 1 and symbol_ratio > 0.05

    def _is_symbol_noise_line(self, line: str) -> bool:
        symbol_count = len(re.findall(r"[^A-Za-z0-9\u4e00-\u9fff\s]", line))
        readable_count = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", line))
        if readable_count == 0 and symbol_count >= 3:
            return True
        symbol_ratio = symbol_count / max(len(line), 1)
        has_natural_phrase = bool(re.search(r"[A-Za-z]{3,}|[\u4e00-\u9fff]{2,}", line))
        return symbol_ratio > 0.45 and not has_natural_phrase

    def _is_short_uninformative_line(self, line: str) -> bool:
        if re.search(r"[A-Za-z\u4e00-\u9fff]{2,}", line):
            return False
        return len(line) <= 4

    def _should_try_ocr(self, cleaned: str, readability: float) -> bool:
        threshold = max(0.0, min(1.0, float(self.config.ocr_trigger_readability)))
        return not cleaned.strip() or (self.config.ocr_enabled and readability < threshold)

    def _looks_like_pdf_binary(self, file_path: Path) -> bool:
        try:
            header = file_path.read_bytes()[:8]
        except Exception:
            return True
        return header.startswith(b"%PDF-")

    def _noise_ratio(self, text: str) -> float:
        lowered = text.lower()
        noise_hits = sum(lowered.count(marker) for marker in self._NOISE_MARKERS)
        return min(1.0, noise_hits / max(len(text) / 80.0, 1.0))

    def _readability_score(self, text: str) -> float:
        if not text.strip():
            return 0.0
        readable = re.findall(
            r"[A-Za-z0-9\u4e00-\u9fff，。！？；：、“”‘’（）()、,.!?;:\-_/ ]",
            text,
        )
        readable_ratio = len(readable) / max(len(text), 1)
        score = 0.75 * readable_ratio + 0.25 * (1.0 - self._noise_ratio(text))
        return max(0.0, min(1.0, score))

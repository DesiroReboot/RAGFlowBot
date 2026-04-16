from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

INPUT_PATH = Path('Eval/golden-set.json')
SHARDS = 5

DOMAIN_ZH = {
    'rolling_stock': '机车车辆',
    'signaling': '信号系统',
    'power_supply': '供电系统',
    'railway_operations': '铁路运营',
    'passenger_service': '客运服务',
    'professional_pdfs': '专业文献',
}

DOC_ID_PATTERN = re.compile(r'^railkb-([a-z_]+)-(?:open-)?\d+$')


def _extract_domain(item: dict[str, Any]) -> str:
    evidence_set = item.get('evidence_set')
    if isinstance(evidence_set, list) and evidence_set:
        first = evidence_set[0]
        if isinstance(first, dict):
            doc_id = str(first.get('doc_id', ''))
            match = DOC_ID_PATTERN.match(doc_id)
            if match:
                return match.group(1)
    return 'railway_operations'


def _rewrite_query(item: dict[str, Any], domain_en: str) -> None:
    query = item.get('query')
    if not isinstance(query, str) or '?' not in query:
        return

    domain_zh = DOMAIN_ZH.get(domain_en, domain_en)
    query_lang = str(item.get('query_lang', 'en'))
    task_type = str(item.get('task_type', 'type_a_fact_qa'))
    answerability = str(item.get('answerability', 'answerable'))

    if query_lang == 'zh':
        if task_type == 'type_a_fact_qa':
            if answerability == 'answerable':
                item['query'] = f'在 RailKB 中，{domain_zh}相关的关键约束是什么？请给出可追溯证据。'
            else:
                item['query'] = f'在 RailKB 中，关于{domain_zh}的该说法是否有证据支持？若无请明确说明无法确认。'
        else:
            item['query'] = f'请基于 RailKB，为{domain_zh}主题提出评估改进建议，并说明如何降低无依据结论风险。'
        return

    if task_type == 'type_a_fact_qa':
        if answerability == 'answerable':
            item['query'] = f'What evidence-backed constraints does RailKB specify for {domain_en}?'
        else:
            item['query'] = (
                f'Does RailKB provide evidence for this claim about {domain_en}? '
                'If not, state uncertainty explicitly.'
            )
    else:
        item['query'] = (
            f'How would you improve evaluation for RailKB {domain_en} queries '
            'while keeping unsupported claims low?'
        )


def _repair_keywords(item: dict[str, Any], domain_en: str) -> None:
    required = item.get('keywords_required')
    if not isinstance(required, list):
        return
    for entry in required:
        if not isinstance(entry, dict):
            continue
        kw = entry.get('keyword')
        if isinstance(kw, str) and '?' in kw:
            entry['keyword'] = domain_en


def _repair_expected(item: dict[str, Any], domain_en: str) -> None:
    expected = item.get('expected')
    if not isinstance(expected, dict):
        return
    facts = expected.get('facts')
    if not isinstance(facts, list):
        return
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        key = fact.get('key')
        if isinstance(key, str) and '?' in key:
            fact['key'] = re.sub(r'\?+', 'core_constraints', key)

        accepted_values = fact.get('accepted_values')
        if not isinstance(accepted_values, list):
            continue

        repaired_values: list[str] = []
        for value in accepted_values:
            if not isinstance(value, str):
                continue
            if '?' in value:
                if value.startswith('RailKB'):
                    repaired_values.append(f'RailKB evidence-backed constraints for {domain_en}')
                else:
                    repaired_values.append(f'Evidence-backed statement grounded in RailKB for {domain_en}')
            else:
                repaired_values.append(value)
        if repaired_values:
            fact['accepted_values'] = repaired_values


def _repair_evidence(item: dict[str, Any], domain_en: str) -> None:
    evidence_set = item.get('evidence_set')
    if not isinstance(evidence_set, list):
        return
    for evidence in evidence_set:
        if not isinstance(evidence, dict):
            continue
        uri = evidence.get('uri')
        if isinstance(uri, str) and '?' in uri:
            evidence['uri'] = re.sub(r'/\?+/', f'/{domain_en}/', uri)

        span_hint = evidence.get('span_hint')
        if isinstance(span_hint, str) and '?' in span_hint:
            evidence['span_hint'] = re.sub(r'\?+', 'core_constraints', span_hint)


def _repair_slices_and_tags(item: dict[str, Any], domain_en: str) -> None:
    slices = item.get('slices')
    if isinstance(slices, dict):
        topic = slices.get('topic')
        if isinstance(topic, str) and '?' in topic:
            slices['topic'] = domain_en

    tags = item.get('tags')
    if isinstance(tags, list):
        item['tags'] = [domain_en if isinstance(tag, str) and '?' in tag else tag for tag in tags]


def repair_item(item: dict[str, Any]) -> None:
    domain_en = _extract_domain(item)
    _rewrite_query(item, domain_en)
    _repair_keywords(item, domain_en)
    _repair_expected(item, domain_en)
    _repair_evidence(item, domain_en)
    _repair_slices_and_tags(item, domain_en)


def main() -> int:
    payload = json.loads(INPUT_PATH.read_text(encoding='utf-8'))
    items = payload.get('items')
    if not isinstance(items, list):
        raise ValueError("Expected 'items' in Eval/golden-set.json")

    for item in items:
        if isinstance(item, dict):
            repair_item(item)

    INPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

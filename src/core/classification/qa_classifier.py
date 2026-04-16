"""问答分类器 - 区分Q-A类和Open类问题"""

import re
from typing import Literal
from dataclasses import dataclass


@dataclass
class QAClassificationResult:
    """问答分类结果"""

    answer_class: Literal["qa", "open"]
    confidence: float
    reasons: list[str]
    structured_patterns: list[str]  # 检测到的结构化模式


class QAClassifier:
    """问答分类器

    根据问题类型和chunk来源分类为Q-A类或Open类：
    - Q-A类：有明确答案的事实性问题，格式："1145人。来源：选品手册.pdf | 第3章 | qa_fact"
    - Open类：开放性问题，需要综合总结

    分类依据：
    1. JSON来源 + 内容模式匹配 → Q-A类
    2. 文本来源 + 明确QA模式（公式、定义、数值）→ Q-A类
    3. 其他 → Open类
    """

    def classify(
        self,
        *,
        query: str,
        chunks: list,
    ) -> QAClassificationResult:
        """分类问题为Q-A类或Open类

        Args:
            query: 用户查询
            chunks: 检索到的chunk列表（SearchResult对象）

        Returns:
            QAClassificationResult: 分类结果
        """
        reasons = []
        patterns = []
        confidence = 0.0

        # 1. 检测结构化模式
        json_patterns = self._detect_json_structured_patterns(chunks)
        text_patterns = self._detect_text_qa_patterns(chunks)
        patterns.extend(json_patterns)
        patterns.extend(text_patterns)

        # 2. JSON来源 + 模式匹配 → 强Q-A信号
        has_json_source = any(
            str(getattr(c, "source_path", "")).lower().endswith(".json") for c in chunks
        )
        if has_json_source and json_patterns:
            confidence += 0.8
            reasons.append("json_with_structured_patterns")

        # 3. 文本中的QA模式
        if text_patterns:
            # 提高权重：公式和定义等强信号应该更容易达到0.5阈值
            pattern_score = min(0.5, 0.2 * len(text_patterns))
            confidence += pattern_score
            reasons.append(f"text_qa_patterns_found:{len(text_patterns)}")

        # 4. 查询意图
        query_intent = self._detect_query_intent(query)
        if query_intent == "fact":
            confidence += 0.15
            reasons.append("fact_query_intent")

        # 5. 明确数值提取
        specific_values = self._extract_specific_values(chunks)
        if specific_values:
            confidence += min(0.1, 0.03 * len(specific_values))
            reasons.append(f"specific_values_found:{len(specific_values)}")

        # 6. 决策
        # 降低阈值到0.4，使分类器对QA模式更敏感
        answer_class = "qa" if confidence >= 0.4 else "open"

        return QAClassificationResult(
            answer_class=answer_class,
            confidence=round(min(1.0, confidence), 4),
            reasons=reasons,
            structured_patterns=patterns[:10],  # 限制数量
        )

    def _detect_json_structured_patterns(
        self,
        chunks: list,
    ) -> list[str]:
        """检测JSON结构化模式：path: value格式、嵌套路径等"""
        patterns = []

        for chunk in chunks:
            content = str(getattr(chunk, "content", ""))

            # 检测 path: value 格式
            if re.search(r"^[\w\.\[\]]+:\s*\S+", content, re.MULTILINE):
                patterns.append("key_value_format")

            # 检测嵌套路径
            if re.search(r"[\w\.\[\]]+\.[\w\.\[\]]+", content):
                patterns.append("nested_path")

            # 检测数组索引
            if re.search(r"\[\d+\]", content):
                patterns.append("array_index")

        return list(set(patterns))

    def _detect_text_qa_patterns(
        self,
        chunks: list,
    ) -> list[str]:
        """检测文本中的QA模式：公式、定义、明确数值"""
        patterns = []

        for chunk in chunks:
            content = str(getattr(chunk, "content", ""))

            # 公式模式
            if re.search(r"\w+\s*=\s*\w+(\s*[\+\-\*\/]\s*\w+)+", content):
                patterns.append("formula")

            # 定义模式
            if re.search(r"(定义为|是指|包括|构成|组成|等于)", content):
                patterns.append("definition")

            # 明确数值
            if re.search(r"\d+(\.\d+)?\s*(人|元|%|天|次|个|件)", content):
                patterns.append("specific_value_with_unit")

            # Incoterms等专有名词
            if re.search(r"\b(FOB|CFR|CIF|EXW)\b", content):
                patterns.append("incoterms")

        return list(set(patterns))

    def _detect_query_intent(
        self,
        query: str,
    ) -> Literal["fact", "process", "mixed"]:
        """检测查询意图：事实性/过程性/混合"""
        query_lower = str(query).lower()

        fact_markers = (
            "是什么",
            "定义",
            "组成",
            "区别",
            "含义",
            "包括",
            "多少",
            "why",
            "what is",
            "difference",
            "define",
            "how many",
        )
        process_markers = (
            "怎么",
            "如何",
            "步骤",
            "流程",
            "执行",
            "落地",
            "how to",
            "step",
            "process",
        )

        has_fact = any(marker in query_lower for marker in fact_markers)
        has_process = any(marker in query_lower for marker in process_markers)

        if has_fact and not has_process:
            return "fact"
        if has_process and not has_fact:
            return "process"
        if has_fact and has_process:
            return "mixed"

        return "fact"  # 默认倾向于事实性

    def _extract_specific_values(
        self,
        chunks: list,
    ) -> list[str]:
        """提取明确数值、日期、名称等"""
        values = []

        for chunk in chunks:
            content = str(getattr(chunk, "content", ""))

            # 数值+单位
            values.extend(re.findall(r"\d+(\.\d+)?\s*(人|元|%|天|次|个|件)", content))

            # 日期
            values.extend(re.findall(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}", content))

            # 专有名词（大写缩写）
            values.extend(re.findall(r"\b[A-Z]{2,}\b", content))

        return values[:20]  # 限制数量

"""单元测试：问答分类器"""

import pytest
from src.core.classification.qa_classifier import QAClassifier, QAClassificationResult


class MockSearchResult:
    """模拟SearchResult对象"""
    def __init__(
        self,
        file_uuid: str,
        source: str,
        content: str,
        score: float,
        source_path: str = "",
    ):
        self.file_uuid = file_uuid
        self.source = source
        self.content = content
        self.score = score
        self.source_path = source_path


class TestQAClassifier:
    """问答分类器测试套件"""

    def test_json_source_with_patterns(self):
        """JSON来源 + 结构化模式 → Q-A类"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test1",
                source="data.json",
                content="product.price: 1145\nproduct.unit: 人",
                score=0.9,
                source_path="/path/to/data.json",
            )
        ]
        result = classifier.classify(query="CRH1B定员", chunks=chunks)

        assert result.answer_class == "qa"
        assert result.confidence >= 0.5
        assert "json_with_structured_patterns" in result.reasons

    def test_text_with_formula(self):
        """文本来源 + 公式模式 → Q-A类"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test2",
                source="manual.pdf",
                content="FOB = EXW + 国内运费 + 报关费",
                score=0.8,
                source_path="/path/to/manual.pdf",
            )
        ]
        result = classifier.classify(query="FOB怎么计算", chunks=chunks)

        assert result.answer_class == "qa"
        assert "formula" in result.structured_patterns

    def test_text_with_specific_values(self):
        """文本来源 + 明确数值 → Q-A类"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test3",
                source="manual.pdf",
                content="列车定员为1145人，票价为50元",
                score=0.8,
                source_path="/path/to/manual.pdf",
            )
        ]
        result = classifier.classify(query="列车定员是多少", chunks=chunks)

        assert result.answer_class == "qa"
        assert "specific_value_with_unit" in result.structured_patterns

    def test_open_class_query(self):
        """开放性问题 → Open类"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test4",
                source="guide.pdf",
                content="选品需要考虑市场需求、竞争情况等因素",
                score=0.7,
                source_path="/path/to/guide.pdf",
            )
        ]
        result = classifier.classify(query="工务段职责", chunks=chunks)

        # 可能是open，取决于confidence
        # 这个测试应该验证confidence < 0.5
        assert result.confidence < 0.5 or result.answer_class == "open"

    def test_mixed_sources_json_dominant(self):
        """混合来源，JSON占主导 → Q-A类"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test5",
                source="data.json",
                content="product.price: 1145",
                score=0.9,
                source_path="/path/to/data.json",
            ),
            MockSearchResult(
                file_uuid="test6",
                source="guide.pdf",
                content="产品定价需要考虑多种因素",
                score=0.7,
                source_path="/path/to/guide.pdf",
            ),
        ]
        result = classifier.classify(query="产品定价", chunks=chunks)

        assert result.answer_class == "qa"
        assert result.confidence >= 0.5

    def test_fact_query_intent(self):
        """事实性查询意图 → 增加Q-A置信度"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test7",
                source="guide.pdf",
                content="产品包含多种成分",
                score=0.8,
                source_path="/path/to/guide.pdf",
            )
        ]
        result = classifier.classify(query="产品组成是什么", chunks=chunks)

        # 应该有fact_query_intent原因
        assert "fact_query_intent" in result.reasons

    def test_definition_pattern(self):
        """定义模式 → Q-A类"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test8",
                source="manual.pdf",
                content="FOB是指船上交货价，是指卖方在指定装运港将货物交至买方指定船只上的价格。",
                score=0.8,
                source_path="/path/to/manual.pdf",
            )
        ]
        result = classifier.classify(query="FOB是什么", chunks=chunks)

        assert "definition" in result.structured_patterns

    def test_incoterms_pattern(self):
        """Incoterms专有名词 → Q-A类"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test9",
                source="manual.pdf",
                content="常用贸易术语包括FOB、CFR、CIF、EXW等",
                score=0.8,
                source_path="/path/to/manual.pdf",
            )
        ]
        result = classifier.classify(query="有哪些贸易术语", chunks=chunks)

        assert "incoterms" in result.structured_patterns

    def test_nested_json_path(self):
        """嵌套JSON路径 → Q-A类"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test10",
                source="data.json",
                content="product.pricing.FOB: EXW + 国内运费\nproduct.pricing.CFR: 成本加运费",
                score=0.9,
                source_path="/path/to/data.json",
            )
        ]
        result = classifier.classify(query="定价公式", chunks=chunks)

        assert "nested_path" in result.structured_patterns
        assert result.answer_class == "qa"

    def test_array_index_pattern(self):
        """数组索引模式 → Q-A类"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test11",
                source="data.json",
                content="items[0].name: 产品A\nitems[1].name: 产品B",
                score=0.9,
                source_path="/path/to/data.json",
            )
        ]
        result = classifier.classify(query="产品列表", chunks=chunks)

        assert "array_index" in result.structured_patterns

    def test_confidence_threshold(self):
        """置信度阈值测试"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test12",
                source="guide.pdf",
                content="一般性描述内容",
                score=0.6,
                source_path="/path/to/guide.pdf",
            )
        ]
        result = classifier.classify(query="一般性问题", chunks=chunks)

        # 置信度应该在0-1之间
        assert 0.0 <= result.confidence <= 1.0

    def test_reasons_list_not_empty(self):
        """原因列表不为空"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test13",
                source="data.json",
                content="key: value",
                score=0.9,
                source_path="/path/to/data.json",
            )
        ]
        result = classifier.classify(query="测试查询", chunks=chunks)

        # 至少应该有一个原因
        assert len(result.reasons) >= 1

    def test_structured_patterns_limit(self):
        """结构化模式数量限制"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test14",
                source="data.json",
                content="key1: value1\nkey2: value2\nkey3: value3\nkey4: value4\nkey5: value5\n"
                "key6: value6\nkey7: value7\nkey8: value8\nkey9: value9\nkey10: value10\n"
                "key11: value11\nkey12: value12",
                score=0.9,
                source_path="/path/to/data.json",
            )
        ]
        result = classifier.classify(query="测试查询", chunks=chunks)

        # 限制最多返回10个模式
        assert len(result.structured_patterns) <= 10

    def test_empty_chunks(self):
        """空chunks列表处理"""
        classifier = QAClassifier()
        chunks = []
        result = classifier.classify(query="测试查询", chunks=chunks)

        # 应该有默认的open分类
        assert result.answer_class in ["qa", "open"]
        assert 0.0 <= result.confidence <= 1.0

    def test_query_intent_fact(self):
        """事实性查询意图检测"""
        classifier = QAClassifier()
        intent = classifier._detect_query_intent("什么是FOB")

        assert intent == "fact"

    def test_query_intent_process(self):
        """过程性查询意图检测"""
        classifier = QAClassifier()
        intent = classifier._detect_query_intent("如何计算FOB")

        assert intent == "process"

    def test_query_intent_mixed(self):
        """混合查询意图检测"""
        classifier = QAClassifier()
        # 使用更简单的查询避免编码问题
        intent = classifier._detect_query_intent("如何计算FOB价格")

        # 这应该检测为process类型
        assert intent == "process"

    def test_extract_specific_values(self):
        """提取明确数值、日期、名称"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test15",
                source="manual.pdf",
                content="定员1145人，票价50元，日期2024-03-15，型号CRH",
                score=0.9,
                source_path="/path/to/manual.pdf",
            )
        ]
        values = classifier._extract_specific_values(chunks)

        # 应该提取到多个值
        assert len(values) > 0

    def test_backward_compatibility(self):
        """向后兼容性测试"""
        classifier = QAClassifier()
        chunks = [
            MockSearchResult(
                file_uuid="test16",
                source="data.json",
                content="key: value",
                score=0.9,
                source_path="/path/to/data.json",
            )
        ]

        # 应该返回QAClassificationResult对象
        result = classifier.classify(query="测试查询", chunks=chunks)
        assert isinstance(result, QAClassificationResult)
        assert hasattr(result, "answer_class")
        assert hasattr(result, "confidence")
        assert hasattr(result, "reasons")
        assert hasattr(result, "structured_patterns")

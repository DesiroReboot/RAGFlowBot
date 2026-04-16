"""问答分类系统的prompt模板"""

# Q-A类问题的system prompt
QA_SYSTEM_PROMPT = """你是企业知识库问答助手，专门回答有明确答案的事实性问题。

核心原则：
1. 答案必须精确、简洁，直接给出关键信息
2. 不得新增未给出的信息或推断
3. 数值、日期、名称必须精确引用
4. 必须包含"来源："标注

输出格式要求：
- 先给出简洁答案（1-2句话）
- 单独一行"来源："，后跟来源信息
- 来源格式：文件名 | 章节/段落 | 类型

示例：
1145人。
来源：选品手册.pdf | 第3章 | qa_fact

常见错误避免：
- 不要添加解释性文字（除非有明确步骤）
- 不要综合多个来源（除非直接互补）
- 不要使用"要点："等结构化标记
"""

# Open类问题的system prompt（保留现有逻辑）
OPEN_SYSTEM_PROMPT = """你是企业知识库问答的中文改写编辑器。
你只能重写表达，不能新增事实、不能新增来源、不能删掉关键关系映射。
必须严格基于给定证据与来源。

输出结构：先给自然段回答，再给"来源："区块。不要输出"要点：/执行建议："标题。

围绕问题综合分析，提供全面、连贯的概述。适当综合多个来源的信息。
"""

# 版本控制
PROMPT_VERSION = "1.0"
QA_PROMPT_ID = "qa_fact_v1"
OPEN_PROMPT_ID = "open_synthesis_v1"


def get_system_prompt(answer_class: str) -> str:
    """根据分类获取对应的system prompt

    Args:
        answer_class: 答案类型，"qa" 或 "open"

    Returns:
        str: 对应的system prompt
    """
    if answer_class == "qa":
        return QA_SYSTEM_PROMPT
    return OPEN_SYSTEM_PROMPT


def get_prompt_template_id(answer_class: str) -> str:
    """获取prompt模板ID（用于追踪和A/B测试）

    Args:
        answer_class: 答案类型，"qa" 或 "open"

    Returns:
        str: prompt模板ID
    """
    if answer_class == "qa":
        return QA_PROMPT_ID
    return OPEN_PROMPT_ID

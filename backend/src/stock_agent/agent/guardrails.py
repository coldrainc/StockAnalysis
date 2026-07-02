from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class GuardrailAction(str, Enum):
    ALLOW = "allow"
    REPAIR = "repair"
    BLOCK = "block"


@dataclass(frozen=True)
class GuardrailFinding:
    code: str
    message: str
    action: GuardrailAction


@dataclass
class GuardrailResult:
    text: str
    findings: list[GuardrailFinding] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(finding.action == GuardrailAction.BLOCK for finding in self.findings)

    @property
    def repaired(self) -> bool:
        return any(finding.action == GuardrailAction.REPAIR for finding in self.findings)

    def extend(self, other: "GuardrailResult") -> None:
        self.text = other.text
        self.findings.extend(other.findings)


class HarnessGuardrails:
    """Input/output guardrails for the stock analysis harness."""

    def __init__(self, max_input_chars: int = 4000, max_output_chars: int = 2200) -> None:
        self.max_input_chars = max_input_chars
        self.max_output_chars = max_output_chars
        self._secret_patterns = [
            re.compile(r"sk-[A-Za-z0-9][A-Za-z0-9_-]{8,}"),
            re.compile(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*['\"]?[\w.\-]{8,}"),
        ]
        self._rubric_patterns = [
            re.compile(r"评分标准[:：]?.*", re.S),
            re.compile(r"(?i)rubric[:：]?.*", re.S),
        ]

    def check_candidate_input(self, text: str) -> GuardrailResult:
        cleaned = text.strip()
        findings: list[GuardrailFinding] = []
        if len(cleaned) > self.max_input_chars:
            cleaned = cleaned[: self.max_input_chars].rstrip()
            findings.append(
                GuardrailFinding(
                    code="input_truncated",
                    message="用户输入过长，已截断后再处理。",
                    action=GuardrailAction.REPAIR,
                )
            )
        redacted, redacted_count = self._redact_secrets(cleaned)
        if redacted_count:
            cleaned = redacted
            findings.append(
                GuardrailFinding(
                    code="input_secret_redacted",
                    message="用户输入疑似包含密钥或 token，已脱敏。",
                    action=GuardrailAction.REPAIR,
                )
            )
        return GuardrailResult(text=cleaned, findings=findings)

    def check_model_output(self, text: str) -> GuardrailResult:
        cleaned = text.strip()
        findings: list[GuardrailFinding] = []
        redacted, redacted_count = self._redact_secrets(cleaned)
        if redacted_count:
            cleaned = redacted
            findings.append(
                GuardrailFinding(
                    code="output_secret_redacted",
                    message="模型输出疑似包含密钥或 token，已脱敏。",
                    action=GuardrailAction.REPAIR,
                )
            )

        cleaned, rubric_removed = self._remove_rubric(cleaned)
        if rubric_removed:
            findings.append(
                GuardrailFinding(
                    code="rubric_removed",
                    message="模型输出疑似泄露评分标准，已移除相关内容。",
                    action=GuardrailAction.REPAIR,
                )
            )

        if len(cleaned) > self.max_output_chars:
            findings.append(
                GuardrailFinding(
                    code="output_long",
                    message="模型输出较长，已保留全文并生成文档页。",
                    action=GuardrailAction.REPAIR,
                )
            )

        if not self._looks_chinese(cleaned):
            cleaned = "请用中文作答。\n\n" + cleaned
            findings.append(
                GuardrailFinding(
                    code="language_repair_prompt",
                    message="输出不是明显中文，已加中文提示。",
                    action=GuardrailAction.REPAIR,
                )
            )

        return GuardrailResult(text=cleaned, findings=findings)

    def blocked_message(self, findings: list[GuardrailFinding]) -> str:
        reasons = "；".join(finding.message for finding in findings)
        return f"我先暂停处理这条输入：{reasons}请换一种方式描述当前分析需求。"

    def _redact_secrets(self, text: str) -> tuple[str, int]:
        count = 0
        result = text
        for pattern in self._secret_patterns:
            result, replacements = pattern.subn("[已脱敏]", result)
            count += replacements
        return result, count

    def _remove_rubric(self, text: str) -> tuple[str, bool]:
        result = text
        removed = False
        for pattern in self._rubric_patterns:
            result, replacements = pattern.subn("（评分标准已隐藏）", result)
            removed = removed or replacements > 0
        return result.strip(), removed

    def _looks_chinese(self, text: str) -> bool:
        if not text:
            return True
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        ascii_letters = len(re.findall(r"[A-Za-z]", text))
        return chinese_chars >= 8 or chinese_chars >= ascii_letters

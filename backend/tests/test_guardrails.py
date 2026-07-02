from stock_agent.agent.agent_loop import AgentLoop
from stock_agent.core.config import StockConfig
from stock_agent.agent.guardrails import HarnessGuardrails
from stock_agent.agent.harness import ScriptedStockHarness


def test_guardrails_redact_user_secret_before_storing() -> None:
    config = StockConfig(focus_areas=["科技"])
    guardrails = HarnessGuardrails()
    loop = AgentLoop(config, ScriptedStockHarness(config, guardrails=guardrails))
    loop.start()

    result = loop.step("请分析 AAPL，api_key=super-secret-token 是误粘贴内容，风险偏好中等。")

    assert result.guardrail_findings
    assert result.state.turns[0].user is not None
    assert "super-secret-token" not in result.state.turns[0].user
    assert "[已脱敏]" in result.state.turns[0].user


def test_guardrails_remove_rubric_from_output() -> None:
    guardrails = HarnessGuardrails()

    result = guardrails.check_model_output("这里是回答。\n\n评分标准：technical_depth 最高权重。")

    assert result.repaired is True
    assert "technical_depth" not in result.text
    assert "评分标准已隐藏" in result.text


def test_guardrails_flags_long_output_without_truncating() -> None:
    guardrails = HarnessGuardrails(max_output_chars=20)
    text = "这是一个非常非常非常非常非常长的中文回答。"

    result = guardrails.check_model_output(text)

    assert result.repaired is True
    assert result.text == text
    assert result.findings[0].code == "output_long"

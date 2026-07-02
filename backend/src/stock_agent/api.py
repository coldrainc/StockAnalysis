"""Compatibility entrypoint for the desktop HTTP API."""

from stock_agent.services.api import (
    AnalysisTrace,
    AnalysisTraceStep,
    ChatDocument,
    ChatResponse,
    MessageRequest,
    SessionRequest,
    app,
    create_app,
    _response,
)

__all__ = [
    "AnalysisTrace",
    "AnalysisTraceStep",
    "ChatDocument",
    "ChatResponse",
    "MessageRequest",
    "SessionRequest",
    "app",
    "create_app",
    "_response",
]

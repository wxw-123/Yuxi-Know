from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.services.chat_history import ChatHistoryStore
from src import knowledge_base
from src.models import select_model
from src.utils.logging_config import logger


chat = APIRouter(prefix="/chat", tags=["chat"])
history_store = ChatHistoryStore()


class ChatRequest(BaseModel):
    """Incoming chat payload."""

    message: str
    session_id: str | None = None
    system_prompt: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    knowledge_base_id: str | None = None
    tools: list[str] = []
    temperature: float | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


def _format_history(session_id: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history_store.load(session_id):
        messages.append({"role": item["role"], "content": item["content"]})
    return messages


async def _build_messages(payload: ChatRequest, session_id: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if payload.system_prompt:
        messages.append({"role": "system", "content": payload.system_prompt})

    messages.extend(_format_history(session_id))

    if payload.knowledge_base_id:
        try:
            docs = await knowledge_base.aquery(payload.message, db_id=payload.knowledge_base_id, top_k=5)
            if docs:
                context = "\n\n".join(str(d.get("text") or d.get("content") or "") for d in docs if isinstance(d, dict))
                context = context[:2000]
                messages.append({"role": "system", "content": f"知识库检索结果:\n{context}"})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to query knowledge base %s: %s", payload.knowledge_base_id, exc)

    messages.append({"role": "user", "content": payload.message})
    return messages


@chat.post("/stream")
async def stream_chat(payload: ChatRequest = Body(...)):
    """Simple streaming chat endpoint with SQLite-based persistence."""

    if not payload.message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    session_id = payload.session_id or str(uuid.uuid4())

    model = select_model(payload.model_provider, payload.model_name)
    messages = await _build_messages(payload, session_id)

    history_store.append(
        session_id,
        role="user",
        content=payload.message,
        metadata={"tools": payload.tools},
    )

    def _stream() -> Any:
        collected = ""
        yield json.dumps({"type": "start", "session_id": session_id}) + "\n"
        try:
            for chunk in model.call(messages, stream=True):
                delta = getattr(chunk, "content", None)
                if isinstance(delta, list):
                    delta = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in delta])
                if delta is None and hasattr(chunk, "delta"):
                    delta = getattr(chunk, "delta", "")
                if delta:
                    collected += delta
                    yield json.dumps({"type": "chunk", "content": delta}) + "\n"

            history_store.append(
                session_id,
                role="assistant",
                content=collected,
                metadata={"tools": payload.tools},
            )
            yield json.dumps({"type": "end", "session_id": session_id}) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.error("Chat stream failed: %s", exc)
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(_stream(), media_type="application/json")

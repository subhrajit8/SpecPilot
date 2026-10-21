from fastapi import UploadFile
from pypdf import PdfReader                               # PDF text extraction
from starlette.concurrency import run_in_threadpool
from langchain_core.messages import HumanMessage
from main import RUN_KEY_PREFIX, app, RUN_RETENTION_SECONDS
from redis.asyncio import Redis
import io                                                 # in-memory byte streams (to read uploaded PDFs)
import json
import asyncio

def thread_config(user_id: str, session_id: str) -> dict:
    """Build the LangGraph config. thread_id = user + session, so each user's history is isolated."""
    return {"configurable": {"thread_id": f"{user_id}:{session_id}"}}

def extract_text(filename: str, data: bytes) -> str:
    """CPU-bound parsing: turn raw file bytes into text (PDF, or plain txt/md)."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))          # open PDF from bytes
        # Join all pages, tagging each with its page number so the LLM can cite pages.
        return "\n\n".join(
            f"[Page {i + 1}]\n{page.extract_text() or ''}"
            for i, page in enumerate(reader.pages)
        )
    return data.decode("utf-8", errors="ignore")      # treat anything else as text

async def read_file(f: UploadFile) -> str:
    """Read an uploaded file without blocking the event loop."""
    data = await f.read()                             # async read of the upload
    # PDF parsing is slow CPU work; running it in a worker thread keeps the event loop free.
    return await run_in_threadpool(extract_text, f.filename, data)

def sse_event(event: str, data: dict) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def run_keys(run_id: str) -> tuple[str, str]:
    """Return Redis keys for a run's event stream and metadata."""
    # Separate metadata from the stream so the run can be authorized before tailing.
    prefix = f"{RUN_KEY_PREFIX}{run_id}"
    return f"{prefix}:events", f"{prefix}:meta"

async def publish_run_event(
    run_id: str,
    event: str,
    data: dict,
    *,
    terminal: bool = False,
) -> None:
    """Append an event and extend retention for both run records."""
    redis_client: Redis = app.state.redis
    stream_key, metadata_key = run_keys(run_id)
    # Group the append, terminal status (when applicable), and TTL updates atomically.
    pipeline = redis_client.pipeline(transaction=True)
    pipeline.xadd(stream_key, {"event": event, "data": json.dumps(data)})
    if terminal:
        pipeline.hset(
            metadata_key,
            "status",
            "failed" if event == "error" else "completed",
        )
    pipeline.expire(stream_key, RUN_RETENTION_SECONDS)
    pipeline.expire(metadata_key, RUN_RETENTION_SECONDS)
    await pipeline.execute()


async def execute_generation(
    run_id: str,
    user_id: str,
    session_id: str,
    instructions: str,
    compliance_text: str,
    requirements_text: str,
) -> None:
    """Run generation independently of clients and publish each update to Redis."""
    try:
        result = {}
        async for update in app.state.graph.astream(
            {
                "mode": "generate",
                "messages": [HumanMessage(content=instructions)],
                "compliance_text": compliance_text,
                "requirements_text": requirements_text,
            },
            config=thread_config(user_id, session_id),
            stream_mode="updates",
        ):
            for node, values in update.items():
                result.update(values)
                # Publish node completion and finished documents as distinct client events.
                await publish_run_event(
                    run_id,
                    "progress",
                    {"node": node, "status": "completed"},
                )
                if "prd" in values:
                    await publish_run_event(
                        run_id,
                        "document",
                        {"type": "prd", "content": values["prd"]},
                    )
                if "tech_doc" in values:
                    await publish_run_event(
                        run_id,
                        "document",
                        {"type": "tech_doc", "content": values["tech_doc"]},
                    )

        # The terminal event carries the final payload and marks the run complete.
        await publish_run_event(
            run_id,
            "complete",
            {
                "session_id": session_id,
                "prd": result["prd"],
                "tech_doc": result["tech_doc"],
            },
            terminal=True,
        )
    except asyncio.CancelledError:
        # Shutdown cancellation must propagate so the lifespan can close cleanly.
        raise
    except Exception as error:
        # Persist the failure so connected and later readers receive the same outcome.
        await publish_run_event(
            run_id,
            "error",
            {"detail": str(error) or "Generation failed."},
            terminal=True,
        )

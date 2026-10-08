import io                                   # in-memory byte streams (to read uploaded PDFs)
import json
import os                                   # read environment variables
from uuid import uuid4                      # generate opaque server-side session ids
from contextlib import asynccontextmanager  # lets us define startup/shutdown logic for the app
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage          # wraps the user's text as a chat message
from langgraph.checkpoint.memory import MemorySaver       # in-memory history (dev only)
from pydantic import BaseModel                            # request-body validation
from pypdf import PdfReader                               # PDF text extraction

from graph import build_graph                             # our LangGraph pipeline (graph.py)

DATABASE_URL = os.getenv("DATABASE_URL")  # e.g. postgresql://user:pass@host:5432/db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup (before `yield`) and once at shutdown (after it)."""
    if DATABASE_URL:
        # Postgres-backed checkpointer: chat history survives restarts and is shared across workers.
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as saver:
            await saver.setup()                       # create the checkpoint tables if missing
            app.state.graph = build_graph(saver)      # build the graph once and share it
            yield                                     # app serves requests while we're paused here
    else:
        # No DB configured: keep history in memory (lost on restart). Development only.
        app.state.graph = build_graph(MemorySaver())
        yield


app = FastAPI(title="Compliance PRD Agent", lifespan=lifespan)  # create the app


def current_user(x_user_id: str = Header(...)) -> str:
    """Return the caller's user id from the X-User-Id header.

    PLACEHOLDER: replace with real JWT/SSO validation in production.
    """
    if not x_user_id.strip():                         # reject empty ids
        raise HTTPException(401, "Missing user")
    return x_user_id.strip()


def thread_config(user_id: str, session_id: str) -> dict:
    """Build the LangGraph config. thread_id = user + session, so each user's history is isolated."""
    return {"configurable": {"thread_id": f"{user_id}:{session_id}"}}


@app.post("/threads", status_code=201)
async def create_thread(user_id: str = Depends(current_user)):
    """Create a new chat thread id on the backend."""
    thread_id = str(uuid4())
    await app.state.graph.aupdate_state(
        thread_config(user_id, thread_id),
        {"mode": "chat"},
    )
    return {"thread_id": thread_id, "session_id": thread_id}


@app.get("/threads")
async def list_threads(user_id: str = Depends(current_user)):
    """List saved chat threads belonging to the authenticated user."""
    thread_prefix = f"{user_id}:"
    latest_by_thread = {}

    async for checkpoint_tuple in app.state.graph.checkpointer.alist(None):
        configurable = checkpoint_tuple.config.get("configurable", {})
        graph_thread_id = configurable.get("thread_id", "")
        if not graph_thread_id.startswith(thread_prefix):
            continue

        checkpoint = checkpoint_tuple.checkpoint
        timestamp = checkpoint.get("ts")
        previous = latest_by_thread.get(graph_thread_id)
        if previous is None or (timestamp or "") > (previous["timestamp"] or ""):
            latest_by_thread[graph_thread_id] = {
                "timestamp": timestamp,
                "channel_values": checkpoint.get("channel_values", {}),
            }

    threads = []
    for graph_thread_id, saved in latest_by_thread.items():
        messages = saved["channel_values"].get("messages", [])
        first_user_message = next(
            (message for message in messages if message.type == "human"),
            None,
        )
        last_message = messages[-1] if messages else None
        title = first_user_message.content if first_user_message else "New chat"
        if not isinstance(title, str):
            title = str(title)

        threads.append({
            "thread_id": graph_thread_id[len(thread_prefix):],
            "session_id": graph_thread_id[len(thread_prefix):],
            "title": title[:120],
            "last_message": last_message.content if last_message else None,
            "updated_at": saved["timestamp"],
        })

    threads.sort(key=lambda thread: thread["updated_at"] or "", reverse=True)
    return {"threads": threads}


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
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(extract_text, f.filename, data)


def sse_event(event: str, data: dict) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/generate")
async def generate(
    session_id: str = Form(...),                                       # Use the id returned by POST /threads
    instructions: str = Form("Generate the PRD and technical documentation."),  # optional PM notes
    compliance_pdf: UploadFile = File(...),                            # compliance guidelines file
    requirements_doc: UploadFile = File(...),                          # client requirements file
    user_id: str = Depends(current_user),                              # authenticated user
):
    compliance_text = await read_file(compliance_pdf)        # extract text from compliance file
    requirements_text = await read_file(requirements_doc)    # extract text from requirements file
    if not compliance_text.strip():                          # scanned PDFs have no text layer
        raise HTTPException(422, "No text extracted from compliance PDF (scanned? run OCR first).")

    async def generate_events():
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
                yield sse_event("progress", {"node": node, "status": "completed"})
                if "prd" in values:
                    yield sse_event("document", {"type": "prd", "content": values["prd"]})
                if "tech_doc" in values:
                    yield sse_event("document", {"type": "tech_doc", "content": values["tech_doc"]})

        yield sse_event("complete", {
            "session_id": session_id,
            "prd": result["prd"],
            "tech_doc": result["tech_doc"],
        })

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ChatIn(BaseModel):
    """JSON body for /chat."""
    session_id: str                                  # which conversation to continue
    message: str                                     # the PM's question or change request


@app.post("/chat")
async def chat(body: ChatIn, user_id: str = Depends(current_user)):
    async def chat_events():
        async for update in app.state.graph.astream(
            {"mode": "chat", "messages": [HumanMessage(content=body.message)]},
            config=thread_config(user_id, body.session_id),
            stream_mode="updates",
        ):
            chat_update = update.get("chat_agent")
            if chat_update and chat_update.get("messages"):
                yield sse_event("reply", {"reply": chat_update["messages"][-1].content})

        yield sse_event("complete", {"session_id": body.session_id})

    return StreamingResponse(
        chat_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/threads/{thread_id}")
async def load_thread(thread_id: str, user_id: str = Depends(current_user)):
    """Load a user's saved conversation and generated documents by thread id."""
    state = await app.state.graph.aget_state(thread_config(user_id, thread_id))
    values = state.values
    if not values:
        raise HTTPException(404, "Thread not found")

    messages = values.get("messages", [])
    if not messages:
        raise HTTPException(404, "Thread not found")

    return {
        "thread_id": thread_id,
        "messages": [
            {"role": "user" if message.type == "human" else "assistant", "content": message.content}
            for message in messages
        ],
        "prd": values.get("prd"),
        "tech_doc": values.get("tech_doc"),
    }


app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent / "frontend", html=True),
    name="frontend",
)
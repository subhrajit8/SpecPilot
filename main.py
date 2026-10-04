import io                                   # in-memory byte streams (to read uploaded PDFs)
import os                                   # read environment variables
from contextlib import asynccontextmanager  # lets us define startup/shutdown logic for the app

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
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


@app.post("/generate")
async def generate(
    session_id: str = Form(...),                                       # The frontend creates the session_id itself when you click New session
    instructions: str = Form("Generate the PRD and technical documentation."),  # optional PM notes
    compliance_pdf: UploadFile = File(...),                            # compliance guidelines file
    requirements_doc: UploadFile = File(...),                          # client requirements file
    user_id: str = Depends(current_user),                              # authenticated user
):
    compliance_text = await read_file(compliance_pdf)        # extract text from compliance file
    requirements_text = await read_file(requirements_doc)    # extract text from requirements file
    if not compliance_text.strip():                          # scanned PDFs have no text layer
        raise HTTPException(422, "No text extracted from compliance PDF (scanned? run OCR first).")

    # ainvoke runs the whole graph asynchronously: every agent awaits its LLM call without blocking.
    result = await app.state.graph.ainvoke(
        {
            "mode": "generate",                                   # tell the router to run the pipeline
            "messages": [HumanMessage(content=instructions)],     # first message in this thread's history
            "compliance_text": compliance_text,
            "requirements_text": requirements_text,
        },
        config=thread_config(user_id, session_id),                # which user's thread to save into
    )
    return {"session_id": session_id, "prd": result["prd"], "tech_doc": result["tech_doc"]}


class ChatIn(BaseModel):
    """JSON body for /chat."""
    session_id: str                                  # which conversation to continue
    message: str                                     # the PM's question or change request


@app.post("/chat")
async def chat(body: ChatIn, user_id: str = Depends(current_user)):
    # mode="chat" routes straight to the chat agent; saved PRD/tech doc/history load from the checkpoint.
    result = await app.state.graph.ainvoke(
        {"mode": "chat", "messages": [HumanMessage(content=body.message)]},
        config=thread_config(user_id, body.session_id),
    )
    return {"reply": result["messages"][-1].content}  # last message = the assistant's answer


@app.get("/history/{session_id}")
async def history(session_id: str, user_id: str = Depends(current_user)):
    # Load the saved state for this user's thread.
    state = await app.state.graph.aget_state(thread_config(user_id, session_id))
    msgs = state.values.get("messages", [])           # empty list if the session doesn't exist
    return [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in msgs
    ]
import asyncio                 
import os                                   # read environment variables
from uuid import UUID, uuid4                # generate opaque server-side ids
from contextlib import asynccontextmanager  # lets us define startup/shutdown logic for the app
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel                            # request-body validation
import json

from utils.authentication import get_current_user
from utils.utility_function import thread_config
from utils.utility_function import read_file
from utils.utility_function import sse_event
from utils.utility_function import run_keys
from utils.utility_function import execute_generation
from main import app, RUN_RETENTION_SECONDS


routes = APIRouter()

@routes.api_route("/health", methods=["GET"])
async def health_check():
    """Health check endpoint to verify that the server is running."""
    return {"status": "ok"}

@routes.api_route("/new_chat", methods=['POST'], status_code=201)
async def create_thread(user_id: str = Depends(get_current_user)):
    """Create a thread and initialize its user-scoped LangGraph checkpoint."""
    session_id = str(uuid4())
    await app.state.graph.aupdate_state(
        thread_config(user_id, session_id),
        {"mode": "generate"},
    )
    return {"thread_id": session_id}


@app.get("/threads")
async def list_threads(user_id: str = Depends(get_current_user)):
    """List saved chat threads belonging to the authenticated user."""
    thread_prefix = f"{user_id}:"
    latest_by_thread = {}

    # Checkpoints are shared by users, so retain only ids with this user's prefix.
    async for checkpoint_tuple in app.state.graph.checkpointer.alist(None):
        configurable = checkpoint_tuple.config.get("configurable", {})
        graph_thread_id = configurable.get("thread_id", "")
        if not graph_thread_id.startswith(thread_prefix):
            continue

        checkpoint = checkpoint_tuple.checkpoint
        timestamp = checkpoint.get("ts")
        previous = latest_by_thread.get(graph_thread_id)
        # A thread can have multiple checkpoints; use its most recently updated one.
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


@app.post("/generate")
async def generate(
    session_id: str = Form(...),                                       # Use the id returned by POST /new_chat
    instructions: str = Form("Generate the PRD and technical documentation."),  # optional PM notes
    compliance_pdf: UploadFile = File(...),                            # compliance guidelines file
    requirements_doc: UploadFile = File(...),                          # client requirements file
    user_id: str = Depends(get_current_user),                              # authenticated user
):
    """Validate uploaded sources, start a background run, and return its identifier."""
    compliance_text = await read_file(compliance_pdf)        # extract text from compliance file
    requirements_text = await read_file(requirements_doc)    # extract text from requirements file
    if not compliance_text.strip():                          # scanned PDFs have no text layer
        raise HTTPException(422, "No text extracted from compliance PDF (scanned? run OCR first).")

    # Persist ownership before scheduling work so the event endpoint can authorize access.
    run_id = str(uuid4())
    _, metadata_key = run_keys(run_id)
    await app.state.redis.hset(
        metadata_key,
        mapping={"user_id": user_id, "session_id": session_id, "status": "running"},
    )
    await app.state.redis.expire(metadata_key, RUN_RETENTION_SECONDS)
    # The worker outlives the HTTP request; its events are the durable client interface.
    task = asyncio.create_task(
        execute_generation(
            run_id,
            user_id,
            session_id,
            instructions,
            compliance_text,
            requirements_text,
        )
    )
    app.state.run_tasks.add(task)
    task.add_done_callback(app.state.run_tasks.discard)
    return {"run_id": run_id, "session_id": session_id}


@app.get("/runs/{run_id}/events")
async def run_events(
    run_id: str,
    request: Request,
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    user_id: str = Depends(get_current_user),
):
    """Tail or replay a run's SSE events, starting after Last-Event-ID when supplied."""
    try:
        # Canonical UUIDs prevent arbitrary input from being used to construct Redis keys.
        run_id = str(UUID(run_id))
    except ValueError as error:
        raise HTTPException(404, "Run not found") from error

    if last_event_id is not None:
        # Redis Stream ids have the form "<milliseconds>-<sequence>".
        parts = last_event_id.split("-", 1)
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise HTTPException(400, "Invalid Last-Event-ID")

    stream_key, metadata_key = run_keys(run_id)
    metadata = await app.state.redis.hgetall(metadata_key)
    # Hide whether another user's run exists by returning the same 404 in both cases.
    if not metadata or metadata.get("user_id") != user_id:
        raise HTTPException(404, "Run not found")

    async def read_events():
        # "0-0" reads from the beginning; a reconnect resumes strictly after its last id.
        cursor = last_event_id or "0-0"
        while not await request.is_disconnected():
            entries = await app.state.redis.xread(
                {stream_key: cursor},
                count=100,
                block=10_000,
            )
            if entries:
                for _, events in entries:
                    for event_id, fields in events:
                        cursor = event_id
                        event = fields["event"]
                        data = json.loads(fields["data"])
                        # The SSE id lets the client request only events it has not received.
                        yield f"id: {event_id}\n{sse_event(event, data)}"
                        if event in {"complete", "error"}:
                            return
            else:
                # Stop if a terminal event was already consumed or the run has otherwise ended.
                if (await app.state.redis.hget(metadata_key, "status")) != "running":
                    return
                # Keep idle connections alive through proxies while waiting for new stream entries.
                yield ": keep-alive\n\n"

    return StreamingResponse(
        read_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


class ChatIn(BaseModel):
    """JSON body for /chat."""
    session_id: str                                  # which conversation to continue
    message: str                                     # the PM's question or change request


@app.post("/chat")
async def chat(body: ChatIn, user_id: str = Depends(get_current_user)):
    """Stream a follow-up answer directly to the current HTTP client."""
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
async def load_thread(thread_id: str, user_id: str = Depends(get_current_user)):
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
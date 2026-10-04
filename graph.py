"""Multi-agent LangGraph pipeline (async).

Flow:  compliance PDF + client requirements
       -> Compliance Extractor -> Requirements Analyst -> PRD Writer -> Tech Doc Writer
The Product Manager reviews the output manually; follow-up edits/questions go through the chat agent.
"""
import os                                              # read environment variables (model name)
from typing import Annotated, Literal, TypedDict       # type hints for the graph state

from langchain_anthropic import ChatAnthropic          # LangChain wrapper around the Claude API
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # chat message types
from langgraph.graph import END, START, StateGraph     # graph builder + special start/end nodes
from langgraph.graph.message import add_messages       # reducer: appends new messages to history

# Model name comes from env var LLM_MODEL; falls back to Claude Sonnet 5.5 if not set.
MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")


def make_llm(temperature: float) -> ChatAnthropic:
    """Create one LLM client with a given temperature (lower = more deterministic)."""
    return ChatAnthropic(
        model=MODEL,            # which Claude model to call
        temperature=temperature,  # randomness of the output
        max_tokens=8000,        # upper limit on tokens in each response
        timeout=180,            # give up on a single API call after 180 seconds
        max_retries=2,          # automatically retry transient API errors twice
    )


# One client per role, each with the temperature that suits its job.
extractor_llm = make_llm(0.0)   # extraction must be faithful and repeatable
analyst_llm = make_llm(0.1)     # analysis: almost deterministic, tiny flexibility
writer_llm = make_llm(0.3)      # PRD/tech doc: natural prose, still grounded
chat_llm = make_llm(0.3)        # follow-up conversation with the PM


class State(TypedDict, total=False):
    """Shared data that flows between agents. total=False means every key is optional."""
    messages: Annotated[list, add_messages]   # chat history; add_messages appends instead of overwriting
    mode: Literal["generate", "chat"]         # which path to run: full pipeline or just chat
    compliance_text: str                      # text extracted from the compliance PDF
    requirements_text: str                    # text extracted from the client requirement doc
    obligations: str                          # output of Compliance Extractor
    requirement_analysis: str                 # output of Requirements Analyst
    prd: str                                  # output of PRD Writer
    tech_doc: str                             # output of Tech Doc Writer


async def _ask(llm: ChatAnthropic, system: str, user: str) -> str:
    """Send one system+user prompt to the LLM and return the reply text.

    async + ainvoke: while we wait for the API, the event loop is FREE to serve other
    requests. (The old llm.invoke() blocked the whole server during the wait.)
    """
    response = await llm.ainvoke(                  # non-blocking call to the model
        [SystemMessage(content=system),            # system message = the agent's role/instructions
         HumanMessage(content=user)]               # user message = the actual input data
    )
    return response.content                        # the generated text


# ---------------------------------------------------------------- Agents (graph nodes)
async def compliance_extractor(state: State):
    """Agent 1: pull every binding obligation out of the compliance guidelines."""
    out = await _ask(
        extractor_llm,                             # temperature 0.0 client
        "You are a fintech compliance analyst. Extract every binding obligation from the "
        "guidelines. Output a markdown table: ID (C-001...), Obligation, Source section/page, "
        "Severity (Mandatory/Recommended), Impacted area (KYC, AML, data privacy, audit, etc.). "
        "Do not invent obligations; quote section numbers exactly as written.",
        state["compliance_text"],                  # the full text of the compliance PDF
    )
    return {"obligations": out}                    # returned keys are merged into the shared State


async def requirements_analyst(state: State):
    """Agent 2: turn client requirements + obligations into structured requirements."""
    out = await _ask(
        analyst_llm,
        "You are a senior business analyst. From the client requirements and the extracted "
        "compliance obligations produce: (1) functional requirements (FR-001...), "
        "(2) non-functional requirements, (3) mapping of each FR to compliance IDs, "
        "(4) conflicts, ambiguities and open questions for the client.",
        # Combine both inputs into one prompt, separated by markdown headings.
        f"# Client requirements\n{state['requirements_text']}\n\n# Obligations\n{state['obligations']}",
    )
    return {"requirement_analysis": out}


async def prd_writer(state: State):
    """Agent 3: write the Product Requirements Document."""
    out = await _ask(
        writer_llm,
        "You are a product manager. Write a complete PRD in markdown: Overview, Goals/Non-goals, "
        "Personas, User stories with acceptance criteria, Functional & Non-functional requirements, "
        "Compliance traceability matrix (requirement -> obligation ID), Risks, Open questions, "
        "Release scope. Every mandatory obligation must be traceable.",
        f"{state['requirement_analysis']}\n\n# Obligations\n{state['obligations']}",
    )
    return {"prd": out}


async def tech_doc_writer(state: State):
    """Agent 4: write the Technical Documentation from the PRD."""
    out = await _ask(
        writer_llm,
        "You are a solution architect. From the PRD write the Technical Documentation in markdown: "
        "Architecture overview, Components, Data model, API contracts, Security & encryption, "
        "Audit logging & retention, Error handling, Integrations, Deployment, Testing strategy, "
        "and compliance control mapping (control -> obligation ID).",
        state["prd"],
    )
    # Add a short assistant message to the saved chat history so the PM sees it in /history.
    summary = AIMessage(
        content="PRD and Technical Documentation generated. Please review them manually; "
                "ask me to refine any section or explain a design decision."
    )
    return {"tech_doc": out, "messages": [summary]}   # add_messages appends the summary to history


async def chat_agent(state: State):
    """Follow-up assistant: answers PM questions using the generated documents as context."""
    # Put the current documents into the system prompt so answers stay grounded in them.
    context = (f"Current PRD:\n{state.get('prd', '(none yet)')}\n\n"
               f"Current Technical Doc:\n{state.get('tech_doc', '(none yet)')}\n\n"
               f"Obligations:\n{state.get('obligations', '(none)')}")
    system = SystemMessage(
        content="You are a PM assistant for a fintech team. Answer using the documents "
                "below. If asked to change something, return the revised section.\n\n" + context
    )
    # system prompt + the whole saved chat history for this user's thread
    reply = await chat_llm.ainvoke([system] + state["messages"])
    return {"messages": [reply]}                   # reply is appended to the saved history


# ---------------------------------------------------------------- Routing
def route_entry(state: State) -> str:
    """Decide where a request starts: full pipeline for 'generate', chat agent otherwise."""
    return "compliance_extractor" if state.get("mode") == "generate" else "chat_agent"


def build_graph(checkpointer):
    """Wire the nodes together and attach the checkpointer (stores per-thread state/history)."""
    g = StateGraph(State)                          # create a graph that uses our State schema

    # Register each function as a named node.
    g.add_node("compliance_extractor", compliance_extractor)
    g.add_node("requirements_analyst", requirements_analyst)
    g.add_node("prd_writer", prd_writer)
    g.add_node("tech_doc_writer", tech_doc_writer)
    g.add_node("chat_agent", chat_agent)

    # From START, pick the path based on state["mode"] (see route_entry).
    g.add_conditional_edges(START, route_entry, ["compliance_extractor", "chat_agent"])

    # Straight-line pipeline: each agent hands its output to the next one.
    g.add_edge("compliance_extractor", "requirements_analyst")
    g.add_edge("requirements_analyst", "prd_writer")
    g.add_edge("prd_writer", "tech_doc_writer")
    g.add_edge("tech_doc_writer", END)             # pipeline finished; PM reviews manually
    g.add_edge("chat_agent", END)                  # a chat turn is one step, then done

    return g.compile(checkpointer=checkpointer)    # checkpointer saves state per thread_id
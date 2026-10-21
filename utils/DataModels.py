from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages       # reducer: appends new messages to history instead of over-writing

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

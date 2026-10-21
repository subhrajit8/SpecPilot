from langchain_anthropic import ChatAnthropic          # LangChain wrapper around the Claude API
from langchain_core.messages import HumanMessage, SystemMessage
import os
    
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

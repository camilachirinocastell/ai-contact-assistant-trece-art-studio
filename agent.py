# agent.py
# Location: ai-contact-assistant-trece-art-studio/agent.py

"""
Defines AgentState (the shared state that flows through the graph), the
graph's nodes, the conditional edge that routes between them, and the
StateGraph itself - the compiled graph that runs the full flow end to end.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END

from config import get_llm
from prompts import (
    ANALYSIS_PROMPT,
    QueryAnalysis,
    PROPOSE_PROMPT,
    REJECT_PROMPT,
    EMAIL_SIGNATURE,
)


class AgentState(TypedDict):
    """
    Shared state that travels through the graph. Every node receives the
    current state and returns a dict with the fields it updated - LangGraph
    merges that into the state before passing it to the next node.
    """

    # The client's original message, exactly as received. Set once at the
    # start of the graph and never modified by any node.
    client_message: str

    # The structured result of analyze_query. Holds service_type,
    # definition_level, urgency and is_valid_project - route_decision reads
    # state["analysis"].is_valid_project directly from here instead of a
    # separate field, so this data has a single source of truth.
    analysis: QueryAnalysis

    # The final client-facing reply, written by whichever node runs after
    # route_decision (propose_node or reject_node). Empty/unset until then.
    response: str


def analyze_query(state: AgentState) -> dict:
    """
    First node of the graph. Calls the LLM with ANALYSIS_PROMPT and forces
    structured output (QueryAnalysis) instead of free text - there is no
    parsing step here because with_structured_output() already returns a
    validated QueryAnalysis instance.
    """
    # with_structured_output() wraps the LLM so its response is coerced into
    # QueryAnalysis. If the model's output doesn't fit the schema, this
    # raises instead of silently returning bad data.
    structured_llm = get_llm().with_structured_output(QueryAnalysis)

    # ANALYSIS_PROMPT is a template (unfilled); the pipe (|) builds a chain
    # that fills it with client_message and sends the result to the LLM.
    chain = ANALYSIS_PROMPT | structured_llm

    result = chain.invoke({"client_message": state["client_message"]})

    # Partial state update: only "analysis" changes here. LangGraph merges
    # this with the rest of the state, so client_message stays untouched.
    return {"analysis": result}


def route_decision(state: AgentState) -> str:
    """
    Conditional edge. Unlike every other function in this file, it never
    calls the LLM - it only reads the analysis analyze_query already
    computed and returns the name of the next node to run.
    """
    if state["analysis"].is_valid_project:
        return "propose_node"
    return "reject_node"


def _extract_text(content) -> str:
    """
    Normalizes the LLM's raw response into plain text. response.content is
    not always a string - sometimes it comes back as a list of content
    blocks - so both nodes below rely on this instead of duplicating the
    same check twice.
    """
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def propose_node(state: AgentState) -> dict:
    """
    Runs when route_decision finds a valid project. Drafts the body of the
    acceptance email and appends the fixed signature - the LLM never writes
    contact details, only the message body.
    """
    chain = PROPOSE_PROMPT | get_llm()
    analysis = state["analysis"]

    result = chain.invoke({
        "client_message": state["client_message"],
        "service_type": analysis.service_type,
        "definition_level": analysis.definition_level,
        "urgency": analysis.urgency,
    })

    body = _extract_text(result.content)
    return {"response": body + EMAIL_SIGNATURE}


def reject_node(state: AgentState) -> dict:
    """
    Runs when route_decision finds an out-of-scope inquiry. Drafts a polite
    decline and appends the same fixed signature as propose_node.
    """
    chain = REJECT_PROMPT | get_llm()
    result = chain.invoke({"client_message": state["client_message"]})

    body = _extract_text(result.content)
    return {"response": body + EMAIL_SIGNATURE}


def build_graph():
    """
    Assembles the graph: registers every node as a station, wires
    route_decision as the sign that routes the parcel after analyze_query,
    and compiles the result into something that can actually be run.
    """
    graph = StateGraph(AgentState)

    graph.add_node("analyze_query", analyze_query)
    graph.add_node("propose_node", propose_node)
    graph.add_node("reject_node", reject_node)

    graph.set_entry_point("analyze_query")

    # The dict maps route_decision's return value to the real node name it
    # points to. Here they happen to match, but LangGraph always needs this
    # mapping made explicit - it never assumes a string == a node name.
    graph.add_conditional_edges(
        "analyze_query",
        route_decision,
        {
            "propose_node": "propose_node",
            "reject_node": "reject_node",
        },
    )

    graph.add_edge("propose_node", END)
    graph.add_edge("reject_node", END)

    return graph.compile()



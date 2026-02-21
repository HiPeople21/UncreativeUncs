"""
LangGraph-powered agentic LinkedIn candidate discovery.

Multi-step agent workflow:
  1. plan_search  → Agent analyzes filters and plans the search strategy
  2. generate     → Agent generates candidate profiles matching the plan
  3. evaluate     → Agent scores and ranks candidates for relevance
  4. decide       → Routes to refine (if quality is low) or finish

Graph:
  plan_search → generate → evaluate → decide → [refine → generate] or [finish]
"""

import hashlib
import json
import re
import time
from typing import Annotated, Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from models import Candidate, SearchFilters

# ── Config ────────────────────────────────────────────────────────────
OLLAMA_MODEL = "qwen2.5"
MAX_REFINE_LOOPS = 1  # At most 1 refinement pass to keep latency reasonable
CACHE_TTL = 300

_cache: dict[str, tuple[float, list[Candidate]]] = {}

# ── LLM instance ─────────────────────────────────────────────────────
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.8,
    num_predict=4096,
)


# ── Agent State ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    """State passed through the LangGraph agent nodes."""
    filters: dict                    # Original search filters
    search_strategy: str             # Plan from the planning node
    raw_candidates: list[dict]       # Generated candidate dicts
    evaluated_candidates: list[dict] # Candidates with scores
    refinement_feedback: str         # Feedback for refinement
    refinement_count: int            # How many times we've refined
    final_candidates: list[dict]     # Final output


# ── Helper functions ──────────────────────────────────────────────────
def _get_initials(name: str) -> str:
    words = name.strip().split()
    if len(words) >= 2:
        return (words[0][0] + words[-1][0]).upper()
    elif words:
        return words[0][0].upper()
    return "?"


def _extract_json(text: str) -> list[dict] | dict:
    """Extract JSON from LLM output (handles markdown fences)."""
    text = text.strip()
    # Remove markdown fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the text
        for pattern in [r"(\[.*\])", r"(\{.*\})"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
    return []


# ── Node 1: Plan Search ──────────────────────────────────────────────
def plan_search(state: AgentState) -> dict:
    """Agent plans the search strategy based on filters."""
    filters = state["filters"]
    skills = ", ".join(filters.get("skills", []))
    level = filters.get("experience_level", "any")
    location = filters.get("location", "any")

    messages = [
        SystemMessage(content=(
            "You are a senior technical recruiter planning a LinkedIn candidate search. "
            "Analyze the search criteria and create a brief search strategy. "
            "Consider: what types of roles match these skills, what companies to target, "
            "what seniority indicators to look for, and how to find diverse candidates. "
            "Keep your response to 3-5 bullet points."
        )),
        HumanMessage(content=(
            f"Plan a LinkedIn search for candidates with:\n"
            f"- Skills: {skills or 'any'}\n"
            f"- Experience level: {level}\n"
            f"- Location: {location}\n"
            f"What's the best strategy to find strong matches?"
        )),
    ]

    response = llm.invoke(messages)
    return {"search_strategy": response.content}


# ── Node 2: Generate Candidates ──────────────────────────────────────
def generate_candidates(state: AgentState) -> dict:
    """Agent generates candidate profiles following the search strategy."""
    filters = state["filters"]
    strategy = state["search_strategy"]
    refinement = state.get("refinement_feedback", "")
    skills = ", ".join(filters.get("skills", []))
    level = filters.get("experience_level", "any")
    location = filters.get("location", "any")

    refinement_note = ""
    if refinement:
        refinement_note = (
            f"\n\nIMPORTANT — Previous results were not good enough. "
            f"Here is the feedback to incorporate:\n{refinement}\n"
            f"Generate DIFFERENT candidates this time."
        )

    messages = [
        SystemMessage(content=(
            "You are a LinkedIn recruitment research agent. Generate realistic candidate "
            "profiles that match the search criteria. You MUST return ONLY a valid JSON "
            "array — no markdown fences, no explanation.\n\n"
            "Each candidate object must have:\n"
            '- "name": Realistic full name (diverse backgrounds)\n'
            '- "headline": LinkedIn headline (role + company)\n'
            '- "location": City or "Remote"\n'
            '- "skills": Array of technical skills\n'
            '- "experience": One of "junior", "mid", "senior", "lead", "principal"\n'
            '- "snippet": 1-2 sentence LinkedIn summary\n'
            '- "company": Current company\n\n'
            "Rules:\n"
            "1. Every candidate MUST have at least one of the requested skills\n"
            "2. Use real companies (Google, Meta, Stripe, etc.) AND startups\n"
            "3. Make names diverse (mix of ethnicities and genders)\n"
            "4. Return ONLY the JSON array"
        )),
        HumanMessage(content=(
            f"Search strategy:\n{strategy}\n\n"
            f"Generate 10 candidates matching:\n"
            f"- Skills: {skills}\n"
            f"- Level: {level}\n"
            f"- Location: {location}"
            f"{refinement_note}"
        )),
    ]

    response = llm.invoke(messages)
    candidates = _extract_json(response.content)

    if isinstance(candidates, dict):
        candidates = [candidates]

    return {"raw_candidates": candidates if isinstance(candidates, list) else []}


# ── Node 3: Evaluate Candidates ──────────────────────────────────────
def evaluate_candidates(state: AgentState) -> dict:
    """Agent evaluates and scores the generated candidates."""
    filters = state["filters"]
    candidates = state["raw_candidates"]
    skills = filters.get("skills", [])

    if not candidates:
        return {"evaluated_candidates": [], "refinement_feedback": "No candidates were generated. Try again with more creative profiles."}

    messages = [
        SystemMessage(content=(
            "You are a recruitment quality analyst. Evaluate each candidate for relevance "
            "to the search criteria. Return ONLY a JSON array where each object has:\n"
            '- "index": The candidate\'s position (0-based) in the input array\n'
            '- "score": Relevance score 1-10\n'
            '- "reason": One-sentence reason for the score\n\n'
            "Score based on:\n"
            "- Skill match (do they have the required skills?)\n"
            "- Experience level match\n"
            "- Location match\n"
            "- Profile quality (realistic headline, good snippet)\n\n"
            "Return ONLY the JSON array."
        )),
        HumanMessage(content=(
            f"Required skills: {', '.join(skills)}\n"
            f"Required level: {filters.get('experience_level', 'any')}\n"
            f"Required location: {filters.get('location', 'any')}\n\n"
            f"Candidates to evaluate:\n{json.dumps(candidates, indent=2)}"
        )),
    ]

    response = llm.invoke(messages)
    evaluations = _extract_json(response.content)

    if not isinstance(evaluations, list):
        evaluations = []

    # Merge evaluations with candidates
    eval_map = {}
    for ev in evaluations:
        if isinstance(ev, dict) and "index" in ev:
            eval_map[ev["index"]] = ev

    evaluated = []
    for i, c in enumerate(candidates):
        ev = eval_map.get(i, {"score": 5, "reason": "Not evaluated"})
        c["_score"] = ev.get("score", 5)
        c["_reason"] = ev.get("reason", "")
        evaluated.append(c)

    # Sort by score descending
    evaluated.sort(key=lambda x: x.get("_score", 0), reverse=True)

    # Check avg quality
    avg_score = sum(c.get("_score", 0) for c in evaluated) / max(len(evaluated), 1)
    feedback = ""
    if avg_score < 5:
        feedback = (
            f"Average quality score was only {avg_score:.1f}/10. "
            f"Candidates don't match the filters well enough. "
            f"Generate candidates with stronger skill matches and more relevant experience."
        )

    return {
        "evaluated_candidates": evaluated,
        "refinement_feedback": feedback,
    }


# ── Node 4: Decide — route to refine or finish ───────────────────────
def decide(state: AgentState) -> Literal["refine", "finish"]:
    """Decide whether to refine results or finish."""
    feedback = state.get("refinement_feedback", "")
    count = state.get("refinement_count", 0)

    if feedback and count < MAX_REFINE_LOOPS:
        return "refine"
    return "finish"


# ── Node 5: Prepare refinement ───────────────────────────────────────
def prepare_refinement(state: AgentState) -> dict:
    """Increment refinement counter before re-generating."""
    return {"refinement_count": state.get("refinement_count", 0) + 1}


# ── Node 6: Finalize ─────────────────────────────────────────────────
def finalize(state: AgentState) -> dict:
    """Package final candidates."""
    return {"final_candidates": state.get("evaluated_candidates", [])}


# ── Build the LangGraph ──────────────────────────────────────────────
def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("plan_search", plan_search)
    graph.add_node("generate", generate_candidates)
    graph.add_node("evaluate", evaluate_candidates)
    graph.add_node("refine", prepare_refinement)
    graph.add_node("finalize", finalize)

    # Set entry point
    graph.set_entry_point("plan_search")

    # Edges
    graph.add_edge("plan_search", "generate")
    graph.add_edge("generate", "evaluate")

    # Conditional: evaluate → decide → refine or finish
    graph.add_conditional_edges(
        "evaluate",
        decide,
        {"refine": "refine", "finish": "finalize"},
    )

    # Refine loops back to generate
    graph.add_edge("refine", "generate")

    # Finalize ends the graph
    graph.add_edge("finalize", END)

    return graph


# Compile once at module level
agent_graph = _build_graph().compile()


# ── Public API ────────────────────────────────────────────────────────
def _to_candidate(data: dict, search_skills: list[str]) -> Candidate:
    """Convert an agent-generated dict into a Candidate."""
    name = data.get("name", "Unknown")
    slug = name.lower().replace(" ", "-").replace("'", "")
    cid = hashlib.md5(f"{name}-{data.get('headline', '')}".encode()).hexdigest()[:12]

    cand_skills = data.get("skills", [])
    matched = [s for s in search_skills if s.lower() in [sk.lower() for sk in cand_skills]]
    if not matched:
        matched = cand_skills[:3]

    return Candidate(
        id=cid,
        name=name,
        headline=data.get("headline", ""),
        location=data.get("location", ""),
        profile_url=f"https://www.linkedin.com/in/{slug}",
        snippet=data.get("snippet", ""),
        matched_skills=matched,
        avatar_initials=_get_initials(name),
    )


async def search_linkedin_profiles(
    filters: SearchFilters,
) -> tuple[list[Candidate], bool]:
    """
    Run the LangGraph agent to discover candidates.
    Returns (candidates, has_more).
    """
    # ── Cache check ───────────────────────────────────────────
    cache_key = f"{sorted(filters.skills)}:{filters.experience_level}:{filters.location}:{filters.page}"
    key = hashlib.md5(cache_key.encode()).hexdigest()

    if key in _cache:
        ts, cached = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return cached, False

    # ── Run the agent graph ───────────────────────────────────
    initial_state: AgentState = {
        "filters": {
            "skills": filters.skills,
            "experience_level": filters.experience_level or "",
            "location": filters.location or "",
        },
        "search_strategy": "",
        "raw_candidates": [],
        "evaluated_candidates": [],
        "refinement_feedback": "",
        "refinement_count": 0,
        "final_candidates": [],
    }

    try:
        result = agent_graph.invoke(initial_state)
        raw_list = result.get("final_candidates", [])

        candidates = []
        for data in raw_list:
            try:
                candidates.append(_to_candidate(data, filters.skills))
            except Exception:
                continue

        if candidates:
            _cache[key] = (time.time(), candidates)

        return candidates, False

    except Exception as e:
        print(f"[LangGraph Agent Error] {e}")
        return [], False

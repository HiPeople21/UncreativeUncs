"""
LangGraph-powered agentic LinkedIn candidate discovery using REAL data.

Uses DuckDuckGo HTML search (no API key, no rate limits) to find actual
LinkedIn profiles, then uses the Ollama LLM agent to enrich and rank them.

Agent graph:
  plan_search → search_sources → enrich → evaluate → decide → [refine] or [finalize]
"""

import datetime
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from github.tools import get_github_tools
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from models import Candidate, SearchFilters

# ── Config ────────────────────────────────────────────────────────────
MODEL_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
MAX_REFINE_LOOPS = 1
CACHE_TTL = 300

_cache: dict[str, tuple[float, list[Candidate]]] = {}

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.7,
    num_predict=4096,
)

DDG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

DDG_MAX_CONCURRENT = 4   # Max concurrent DDGS search calls

SOURCE_QUERY_TEMPLATES = [
    "linkedin.com/in {terms}",
]

# Discovery templates
SCHOLAR_DISCOVERY_TEMPLATE = "scholar.google.com {terms}"

MIN_SCHOLAR_CITATIONS = 100
MIN_GITHUB_CONTRIBUTIONS_LAST_YEAR = 100
MIN_GITHUB_PUBLIC_REPOS = 5
MIN_REPO_CONTRIBUTIONS = 10   # Min commits to repo to qualify as significant contributor

# GitHub repo impact thresholds — only crawl large, community-scale projects
MIN_REPO_STARS = 500
MIN_REPO_FORKS = 50
MIN_REPO_TOTAL_CONTRIBUTORS = 10  # Must be a multi-contributor community project

# Recency gate — candidates with no activity in this many years are skipped
RECENCY_CUTOFF_YEARS = 2

# Output filtering — only return the top N most qualified candidates
TOP_CANDIDATES_LIMIT = 20   # Max candidates returned to the recruiter
MIN_CANDIDATE_SCORE = 5     # Drop candidates scored below this threshold
ENRICH_BATCH_SIZE = 25      # Process at most this many candidates per LLM call

_scholar_quality_cache: dict[str, bool] = {}
_github_quality_cache: dict[str, bool] = {}


# ── Agent State ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    filters: dict
    search_query: str
    raw_results: list[dict]          # Raw DuckDuckGo search results
    parsed_candidates: list[dict]    # Parsed from search results
    enriched_candidates: list[dict]  # LLM-enriched candidates
    evaluated_candidates: list[dict] # Scored and ranked
    refinement_feedback: str
    refinement_count: int
    final_candidates: list[dict]


# ── Helpers ───────────────────────────────────────────────────────────
def _get_initials(name: str) -> str:
    words = name.strip().split()
    if len(words) >= 2:
        return (words[0][0] + words[-1][0]).upper()
    elif words:
        return words[0][0].upper()
    return "?"


def _extract_json(text: str) -> list[dict] | dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for pattern in [r"(\[.*\])", r"(\{.*\})"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
    return []


def _search_ddg(query: str) -> list[dict]:
    """Search DuckDuckGo via the duckduckgo_search library (DDGS).

    Uses DDG's API endpoints rather than the HTML scraper, so it is far more
    resilient to rate-limiting and IP blocks.  Each call creates its own DDGS
    session so concurrent threads don't share state.
    """
    try:
        with DDGS() as ddgs:
            raw = ddgs.text(query, max_results=10)
            results = list(raw) if raw else []
        return [
            {
                "title": r.get("title", ""),
                "href": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception as e:
        print(f"[DDG] Search failed for '{query[:60]}': {e}")
        return []


def _canonicalize_url(url: str) -> str:
    """Normalize profile URLs to improve deduplication across queries."""
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")

    # Remove tracking/fragments and default to https key for dedupe only
    canonical = f"{host}{path}"
    return canonical


def _detect_source(url: str) -> str | None:
    lowered = url.lower()
    parsed = urlparse(url)
    if "linkedin.com/in/" in lowered:
        return "LinkedIn"
    if "scholar.google.com/citations" in lowered:
        return "Google Scholar"
    if "github.com/" in lowered:
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        # Keep only user profile pages, not repositories or org sub-pages.
        if len(parts) == 1:
            blocked = {
                "features", "topics", "collections", "trending", "marketplace",
                "sponsors", "about", "pricing", "login", "join", "orgs",
                "organizations", "enterprise", "site", "events", "settings",
                "apps", "search", "explore", "pulls", "issues", "new",
            }
            if parts[0].lower() not in blocked:
                return "GitHub"
    return None


def _extract_first_int(text: str, pattern: str) -> int:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return 0
    try:
        return int(match.group(1).replace(",", ""))
    except Exception:
        return 0


def _is_recent_enough(date_str: str | None) -> bool:
    """Return True if the ISO-8601 date is within RECENCY_CUTOFF_YEARS of today."""
    if not date_str:
        return False
    try:
        dt = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=365 * RECENCY_CUTOFF_YEARS
        )
        return dt >= cutoff
    except Exception:
        return False


def _skills_to_github_tags(skills: list[str]) -> list[str]:
    """Convert skill names to lowercase, hyphenated GitHub topic tags."""
    tags = []
    for skill in skills:
        tag = skill.lower().strip().replace(" ", "-")
        if tag:
            tags.append(tag)
    return tags[:5]


def _is_substantial_scholar_contributor(url: str, title: str, snippet: str) -> bool:
    """Return True if the Scholar profile has enough citations AND recent publications.

    Speed note: we first try to satisfy both checks from the snippet alone.
    The page is only fetched when the snippet is insufficient and we still need
    to verify citations or recency.
    """
    key = _canonicalize_url(url)
    if key in _scholar_quality_cache:
        return _scholar_quality_cache[key]

    parsed = urlparse(url)
    if "user=" not in parsed.query:
        _scholar_quality_cache[key] = False
        return False

    combined = f"{title} {snippet}"
    citations = _extract_first_int(combined, r"cited by\s*([0-9,]+)")

    # Check for recent year in snippet (fast path — avoids a page fetch)
    current_year = datetime.datetime.now().year
    recent_years = {str(y) for y in range(current_year - 2, current_year + 1)}
    has_recent_snippet = any(y in combined for y in recent_years)

    # Only fetch the page if we couldn't determine both signals from the snippet
    if citations == 0 or not has_recent_snippet:
        try:
            resp = httpx.get(url, headers=DDG_HEADERS, follow_redirects=True, timeout=10.0)
            if resp.status_code < 400:
                page = resp.text
                if citations == 0:
                    citations = _extract_first_int(page, r"Cited by\s*([0-9,]+)")
                if not has_recent_snippet:
                    has_recent_snippet = any(y in page for y in recent_years)
        except Exception:
            pass

    # Require significant citations AND at least one recent-year publication
    ok = citations >= MIN_SCHOLAR_CITATIONS and has_recent_snippet
    _scholar_quality_cache[key] = ok
    return ok


def _is_substantial_github_contributor(url: str) -> bool:
    key = _canonicalize_url(url)
    if key in _github_quality_cache:
        return _github_quality_cache[key]

    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) != 1:
        _github_quality_cache[key] = False
        return False
    username = parts[0]

    contributions = 0
    public_repos = 0
    is_user = False

    try:
        profile_resp = httpx.get(url, headers=DDG_HEADERS, follow_redirects=True, timeout=12.0)
        if profile_resp.status_code < 400:
            contributions = _extract_first_int(
                profile_resp.text,
                r'([0-9,]+)\s+contributions?\s+in\s+the\s+last\s+year',
            )
    except Exception:
        contributions = 0

    try:
        api_resp = httpx.get(
            f"https://api.github.com/users/{username}",
            headers={"Accept": "application/vnd.github+json", "User-Agent": DDG_HEADERS["User-Agent"]},
            timeout=12.0,
        )
        if api_resp.status_code < 400:
            data = api_resp.json()
            public_repos = int(data.get("public_repos", 0) or 0)
            is_user = (data.get("type", "") == "User")
    except Exception:
        public_repos = 0
        is_user = False

    ok = is_user and (
        contributions >= MIN_GITHUB_CONTRIBUTIONS_LAST_YEAR
        or public_repos >= MIN_GITHUB_PUBLIC_REPOS
    )
    _github_quality_cache[key] = ok
    return ok


def _extract_name_from_result(url: str, title: str) -> str:
    title = re.sub(r"\s*[\-|:]\s*(LinkedIn|Google Scholar|GitHub).*$", "", title).strip()
    if title and title.lower() not in {
        "linkedin", "google scholar", "github"
    }:
        return title.split(" - ")[0].strip()

    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return "Unknown"

    handle = parts[-1]
    handle = re.sub(r"[_\-]+", " ", handle).strip()
    return handle.title() if handle else "Unknown"


def _parse_candidate_result(result: dict, search_skills: list[str]) -> dict | None:
    """Parse a DDG search result into a cross-source candidate dict."""
    url = result.get("href", "")
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    source = _detect_source(url)

    if not source:
        return None

    # Parse title where possible
    title_cleaned = re.sub(r"\s*\|\s*.*$", "", title).strip()
    parts = [p.strip() for p in title_cleaned.split(" - ", 2)]
    name = _extract_name_from_result(url, title_cleaned)
    headline = " - ".join(parts[1:]) if len(parts) > 1 else ""

    if not name or name.lower() in ["sign in", "log in", ""]:
        return None

    # Extract location from snippet
    location = ""
    loc_patterns = [
        r"(?:Location|Area|Region|Based in)[:\s]+([^·\n.]+)",
        r"(?:located in|based in)\s+([^·\n.]+)",
    ]
    for pattern in loc_patterns:
        match = re.search(pattern, snippet, re.IGNORECASE)
        if match:
            location = match.group(1).strip()[:50]
            break

    # Match skills
    full_text = f"{title} {snippet}".lower()
    matched = [s for s in search_skills if s.lower() in full_text]

    if source == "Google Scholar" and not _is_substantial_scholar_contributor(url, title, snippet):
        return None
    if source == "GitHub" and not _is_substantial_github_contributor(url):
        return None

    return {
        "name": name,
        "headline": f"[{source}] {headline or 'Profile'}",
        "location": location,
        "profile_url": url,
        "snippet": snippet[:300],
        "matched_skills": matched,
    }


def _extract_authors_from_scholar_result(result: dict) -> list[str]:
    """Extract author names from a Google Scholar search result snippet.

    Scholar snippets typically look like:
      "J Smith, A Jones - Nature, 2022 - Cited by 120"
    """
    snippet = result.get("snippet", "")
    authors: list[str] = []
    patterns = [
        # "Firstname Lastname, Firstname Lastname - Journal"
        r"^([A-Z][a-zA-Z\-']+ [A-Z][a-zA-Z\-']+"
        r"(?:,\s*[A-Z][a-zA-Z\-']+ [A-Z][a-zA-Z\-']+)*)\s*[-–]",
        # "… Firstname Lastname, Firstname Lastname - 2022"
        r"([A-Z][a-zA-Z\-']+ [A-Z][a-zA-Z\-']+"
        r"(?:,\s*[A-Z][a-zA-Z\-']+ [A-Z][a-zA-Z\-']+)*)\s*[-–]\s*\d{4}",
    ]
    for pattern in patterns:
        match = re.search(pattern, snippet)
        if match:
            for part in match.group(1).split(","):
                name = part.strip()
                if len(name.split()) >= 2:
                    authors.append(name)
            break
    return authors[:3]  # At most 3 authors per paper


_github_user_cache: dict[str, dict] = {}  # username → API user data


def _fetch_github_user(username: str) -> dict:
    """Fetch and cache GitHub user API data for a given username."""
    if username in _github_user_cache:
        return _github_user_cache[username]
    try:
        resp = httpx.get(
            f"https://api.github.com/users/{username}",
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": DDG_HEADERS["User-Agent"],
            },
            timeout=8.0,
        )
        data = resp.json() if resp.status_code < 400 else {}
    except Exception:
        data = {}
    _github_user_cache[username] = data
    return data


def _process_repo_contributors(owner: str, repo_name: str, repo_html_url: str) -> list[dict]:
    """Fetch and enrich contributors for a repo using GitHubTools.

    Uses the existing GitHubTools.get_contributors() call (authenticated, paginated)
    then enriches each user profile via _fetch_github_user to get name/location/website.
    Filters out bots, orgs, and stale accounts.
    """
    github_tools = get_github_tools()
    contribs_response = github_tools.get_contributors(owner, repo_name, max_results=30)
    if not contribs_response.success:
        print(f"[GitHub] Contributors fetch failed for {owner}/{repo_name}: {contribs_response.error}")
        return []

    all_contribs = contribs_response.contributors
    if len(all_contribs) < MIN_REPO_TOTAL_CONTRIBUTORS:
        print(f"[GitHub] Skip {owner}/{repo_name}: only {len(all_contribs)} contributors")
        return []

    # Keep only substantial contributors (above commit threshold)
    qualified = [c for c in all_contribs if c.contributions >= MIN_REPO_CONTRIBUTIONS]

    def _enrich(contrib) -> dict | None:
        ud = _fetch_github_user(contrib.login)
        if not ud or ud.get("type", "") != "User":
            return None
        if not _is_recent_enough(ud.get("updated_at")):
            print(f"[GitHub] Skip {contrib.login}: stale account")
            return None
        blog = (ud.get("blog") or "").strip()
        website = blog if blog.startswith("http") else (f"https://{blog}" if blog else "")
        return {
            "login": contrib.login,
            "contributions": contrib.contributions,
            "name": ud.get("name") or contrib.login,
            "location": ud.get("location") or "",
            "website": website,
        }

    enriched: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(8, len(qualified))) as executor:
        futures = {executor.submit(_enrich, c): c for c in qualified}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    enriched.append(result)
            except Exception:
                pass

    return enriched


def _contributors_to_candidates(
    contributors: list[dict],
    repo_url: str,
    skills: list[str],
) -> list[dict]:
    """Convert GitHub contributor records into candidate dicts.

    If the contributor has a personal website it is returned as the primary
    profile URL (fulfilling the "personal website instead of LinkedIn" ask).
    The GitHub profile is always included as a second entry so the recruiter
    has both links available.
    """
    candidates: list[dict] = []
    repo_name = "/".join(repo_url.split("github.com/")[-1].split("/")[:2])

    for contrib in contributors:
        username = contrib.get("login", "")
        name = contrib.get("name") or username
        github_url = f"https://github.com/{username}"
        website = contrib.get("website", "")
        location = contrib.get("location", "")
        n_contribs = contrib.get("contributions", 0)
        snippet = f"Contributor to {repo_name} with {n_contribs} commits."

        if website:
            candidates.append({
                "name": name,
                "headline": f"[Personal Website] Contributor to {repo_name} · GitHub: {username}",
                "location": location,
                "profile_url": website,
                "snippet": snippet,
                "matched_skills": skills,
            })

        candidates.append({
            "name": name,
            "headline": f"[GitHub] Contributor to {repo_name}",
            "location": location,
            "profile_url": github_url,
            "snippet": snippet,
            "matched_skills": skills,
        })

    return candidates


# ══════════════════════════════════════════════════════════════════════
# LangGraph Agent Nodes
# ══════════════════════════════════════════════════════════════════════

def _build_linkedin_queries(terms: str, filters: dict) -> list[str]:
    """Generate a diverse set of LinkedIn DDG queries for broad candidate coverage.

    Strategy: vary skill combinations, add role-title and seniority suffixes,
    include/omit location, and use both 'site:' and plain-keyword forms so DDG
    returns different result pages.  Caps at 16 queries for maximum breadth.
    """
    skills = filters.get("skills", [])
    location = filters.get("location", "").strip()

    # Build up to 5 skill phrase bases (most important combinations first)
    bases: list[str] = []
    if terms:
        bases.append(terms)
    if len(skills) >= 2:
        pair = f"{skills[0]} {skills[1]}"
        if pair.lower() != terms.lower():
            bases.append(pair)
    if skills:
        solo = skills[0]
        if solo.lower() not in {b.lower() for b in bases}:
            bases.append(solo)
    if len(skills) >= 3:
        alt = f"{skills[0]} {skills[2]}"
        if alt.lower() not in {b.lower() for b in bases}:
            bases.append(alt)
    if len(skills) >= 3:
        alt2 = f"{skills[1]} {skills[2]}"
        if alt2.lower() not in {b.lower() for b in bases}:
            bases.append(alt2)

    # Location variants: full string and city-only
    loc_variants: list[str] = []
    if location:
        loc_variants.append(location)
        city = location.split(",")[0].strip()
        if city and city.lower() != location.lower():
            loc_variants.append(city)

    role_suffixes = ["engineer", "developer", "researcher", "scientist"]
    seniority_prefixes = ["senior", "staff"]

    queries: list[str] = []

    for base in bases[:4]:
        if loc_variants:
            for loc in loc_variants[:2]:
                queries.append(f"site:linkedin.com/in {base} {loc}")
                queries.append(f"site:linkedin.com/in {base} {role_suffixes[0]} {loc}")
                queries.append(f"site:linkedin.com/in {base} {role_suffixes[1]} {loc}")
                queries.append(f"site:linkedin.com/in {seniority_prefixes[0]} {base} {loc}")
        else:
            queries.append(f"site:linkedin.com/in {base}")
            queries.append(f"site:linkedin.com/in {base} {role_suffixes[0]}")
            queries.append(f"site:linkedin.com/in {base} {role_suffixes[1]}")
            queries.append(f"site:linkedin.com/in {seniority_prefixes[0]} {base}")

    # Plain-keyword forms (no site: operator) — DDG returns different result pages
    if bases:
        plain_loc = f" {loc_variants[0]}" if loc_variants else ""
        queries.append(f"linkedin.com/in {bases[0]}{plain_loc}")
        if len(bases) > 1:
            queries.append(f"linkedin.com/in {bases[1]}{plain_loc}")
        # Remote candidates — different pool from location-specific results
        queries.append(f"site:linkedin.com/in {bases[0]} remote")

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        k = q.lower().strip()
        if k not in seen:
            seen.add(k)
            deduped.append(q)

    return deduped[:16]


def plan_search(state: AgentState) -> dict:
    """LLM plans the core skill/domain search terms.

    Location and experience level are intentionally excluded from the planned
    terms — location is injected separately into LinkedIn queries by
    _build_linkedin_queries, and we search all career stages so the evaluator
    can rank by relevance rather than hard-filtering early.
    """
    filters = state["filters"]
    skills = ", ".join(filters.get("skills", []))
    refinement = state.get("refinement_feedback", "")

    refinement_note = ""
    if refinement:
        refinement_note = f"\n\nPrevious search had issues: {refinement}\nBroaden or adjust accordingly."

    messages = [
        SystemMessage(content=(
            "You are a recruitment search specialist. Produce compact skill/domain "
            "search terms to find candidates across LinkedIn, Google Scholar, and GitHub.\n\n"
            "Rules:\n"
            "1. Focus on the 1–3 most important skills or technologies\n"
            "2. Do NOT include experience level (junior/senior/etc.) — we want all levels\n"
            "3. Do NOT include location — location is handled separately\n"
            "4. Return ONLY the search terms string — nothing else\n"
            "5. Keep it concise: 2–6 words\n\n"
            "Example: 'python machine learning' or 'react typescript frontend'"
        )),
        HumanMessage(content=(
            f"Skills: {skills or 'any'}"
            f"{refinement_note}"
        )),
    ]

    response = llm.invoke(messages)
    query = response.content.strip().strip('"').strip("'").strip("`").splitlines()[0].strip()

    print(f"[Agent] Planned terms: {query}")
    return {"search_query": query}


def search_sources(state: AgentState) -> dict:
    """Two-phase candidate discovery across LinkedIn, Scholar, and GitHub.

    Phase 1 – ALL searches run in parallel:
      • LinkedIn:      multiple diverse queries via _build_linkedin_queries
      • Scholar:       paper/article discovery search
      • GitHub:        repository discovery search

    Phase 2 – contributor/author lookups (also parallel):
      • Scholar papers → extract authors → search their citation profiles
      • GitHub repos   → GitHub API contributors → enrich with personal websites

    Speed optimisations:
      • LinkedIn queries run concurrently via DDGS (up to DDG_MAX_CONCURRENT at once)
      • GitHub repo discovery uses impact-ranked GitHub API search (no DDG needed)
      • GitHub user-profile fetches inside _process_repo_contributors are parallel
      • GitHub user data is cached (_github_user_cache) across repos
      • Scholar citation check uses snippet fast-path; page fetch only if needed
    """
    base_query = state["search_query"]
    filters = state["filters"]
    skills = filters.get("skills", [])

    terms = base_query.strip() or "software engineer"

    # Build the Phase 1 query map (LinkedIn + Scholar via DDG; GitHub via API below)
    linkedin_queries = _build_linkedin_queries(terms, filters)
    scholar_query = SCHOLAR_DISCOVERY_TEMPLATE.format(terms=terms)

    phase1_map: dict[str, str] = {}
    for i, q in enumerate(linkedin_queries):
        phase1_map[f"linkedin_{i}"] = q
    phase1_map["scholar_papers"] = scholar_query

    # ── Phase 1: LinkedIn + Scholar DDG searches in parallel ─────────
    phase1_results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=min(DDG_MAX_CONCURRENT, len(phase1_map))) as executor:
        future_map = {executor.submit(_search_ddg, q): key for key, q in phase1_map.items()}
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                phase1_results[key] = future.result()
            except Exception as e:
                print(f"[DDG] Phase 1 query failed ({key}): {e}")
                phase1_results[key] = []

    # Collect all LinkedIn DDG results
    all_linkedin_raw: list[dict] = []
    for key, results in phase1_results.items():
        if key.startswith("linkedin_"):
            all_linkedin_raw.extend(results)

    scholar_paper_raw = phase1_results.get("scholar_papers", [])
    all_raw_results = all_linkedin_raw + scholar_paper_raw

    # ── Extract author names from Scholar discovery results ───────────
    author_names: list[str] = []
    for result in scholar_paper_raw:
        author_names.extend(_extract_authors_from_scholar_result(result))
    seen_names: set[str] = set()
    unique_authors: list[str] = []
    for a in author_names:
        if a not in seen_names:
            seen_names.add(a)
            unique_authors.append(a)
    unique_authors = unique_authors[:10]  # Cap at 10 for broader author coverage

    # ── Phase 2a: GitHub repos via GitHub API (impact-ranked) ────────
    # Uses search_repos_by_impact to get top community repos for the skills,
    # then fetches contributors for each via GitHubTools.get_contributors().
    github_candidates: list[dict] = []
    tags = _skills_to_github_tags(skills)
    if tags:
        try:
            gh = get_github_tools()
            repo_response = gh.search_repos_by_impact(
                tags=tags,
                min_stars=MIN_REPO_STARS,
                match_all_tags=False,   # OR logic: repos matching ANY skill tag
                max_results=10,
            )
            if repo_response.success:
                print(f"[GitHub] Found {repo_response.total_found} impact-ranked repos for tags {tags}")
                for repo_dict in repo_response.repositories[:5]:
                    owner = (repo_dict.get("owner") or {}).get("login", "")
                    repo_name = repo_dict.get("name", "")
                    repo_html_url = repo_dict.get("html_url", f"https://github.com/{owner}/{repo_name}")
                    if not owner or not repo_name:
                        continue
                    stars = repo_dict.get("stargazers_count", 0)
                    forks = repo_dict.get("forks_count", 0)
                    if forks < MIN_REPO_FORKS:
                        print(f"[GitHub] Skip {owner}/{repo_name}: {forks} forks < {MIN_REPO_FORKS}")
                        continue
                    print(f"[GitHub] Processing {owner}/{repo_name} ({stars}⭐, {forks} forks)")
                    contribs = _process_repo_contributors(owner, repo_name, repo_html_url)
                    github_candidates.extend(_contributors_to_candidates(contribs, repo_html_url, skills))
            else:
                print(f"[GitHub] Repo search failed: {repo_response.error}")
        except Exception as e:
            print(f"[GitHub] Repo discovery error: {e}")

    # ── Phase 2b: Scholar author names → citation profile lookups ─────
    scholar_candidates: list[dict] = []
    if unique_authors:
        with ThreadPoolExecutor(max_workers=min(DDG_MAX_CONCURRENT, len(unique_authors))) as executor:
            future_map = {
                executor.submit(_search_ddg, f"scholar.google.com/citations {author}"): author
                for author in unique_authors
            }
            for future in as_completed(future_map):
                author = future_map[future]
                try:
                    results = future.result()
                    for result in results:
                        url = result.get("href", "")
                        if "scholar.google.com/citations" in url and "user=" in url:
                            parsed = _parse_candidate_result(result, skills)
                            if parsed:
                                scholar_candidates.append(parsed)
                except Exception as e:
                    print(f"[Scholar] Author lookup failed ({author}): {e}")

    # ── Parse all LinkedIn results (deduplicated across queries) ───────
    linkedin_candidates: list[dict] = []
    seen_li: set[str] = set()
    for r in all_linkedin_raw:
        parsed = _parse_candidate_result(r, skills)
        if not parsed:
            continue
        key = _canonicalize_url(parsed.get("profile_url", ""))
        if key and key not in seen_li:
            seen_li.add(key)
            linkedin_candidates.append(parsed)

    # ── Deduplicate and merge all sources ─────────────────────────────
    candidates: list[dict] = []
    seen_urls: set[str] = set(seen_li)  # LinkedIn already deduped above
    candidates.extend(linkedin_candidates)
    for c in scholar_candidates + github_candidates:
        key = _canonicalize_url(c.get("profile_url", ""))
        if key and key not in seen_urls:
            seen_urls.add(key)
            candidates.append(c)

    print(
        f"[Agent] {len(candidates)} unique profiles "
        f"(LinkedIn: {len(linkedin_candidates)} from {len(linkedin_queries)} queries, "
        f"Scholar: {len(scholar_candidates)}, "
        f"GitHub/Personal: {len(github_candidates)})"
    )

    return {
        "raw_results": all_raw_results,
        "parsed_candidates": candidates,
    }


def enrich_candidates(state: AgentState) -> dict:
    """LLM cleans and enriches the real candidate data, in batches for large pools."""
    candidates = state["parsed_candidates"]
    filters = state["filters"]

    if not candidates:
        return {"enriched_candidates": []}

    system_prompt = (
        "You are a recruitment data analyst. Clean and enrich these REAL candidate "
        "search results. For each candidate, return a JSON object with:\n"
        '- "name": Cleaned name\n'
        '- "headline": Cleaned role/title\n'
        '- "location": Location if available, else "Unknown"\n'
        '- "profile_url": Keep the original profile URL exactly as-is\n'
        '- "snippet": Cleaned snippet\n'
        '- "matched_skills": Skills from the search that appear in their profile\n'
        '- "experience_level": Infer from title (junior/mid/senior/lead/principal)\n\n'
        "Return ONLY a JSON array. Do NOT invent information."
    )

    all_enriched: list[dict] = []
    for batch_start in range(0, len(candidates), ENRICH_BATCH_SIZE):
        batch = candidates[batch_start : batch_start + ENRICH_BATCH_SIZE]
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Filters: skills={filters.get('skills')}, "
                f"level={filters.get('experience_level')}, "
                f"location={filters.get('location')}\n\n"
                f"Candidates:\n{json.dumps(batch, indent=2)}"
            )),
        ]
        response = llm.invoke(messages)
        enriched = _extract_json(response.content)
        if isinstance(enriched, dict):
            enriched = [enriched]

        if isinstance(enriched, list):
            # Preserve original URLs in case LLM strips them
            for i, e in enumerate(enriched):
                orig_idx = batch_start + i
                if orig_idx < len(candidates) and not e.get("profile_url"):
                    e["profile_url"] = candidates[orig_idx].get("profile_url", "")
            all_enriched.extend(enriched)
        else:
            all_enriched.extend(batch)  # Fallback: keep originals unchanged

    return {"enriched_candidates": all_enriched}


def evaluate_candidates(state: AgentState) -> dict:
    """LLM scores and ranks candidates by relevance, in batches for large pools."""
    candidates = state["enriched_candidates"]
    filters = state["filters"]

    if not candidates:
        return {
            "evaluated_candidates": [],
            "refinement_feedback": "No candidate profiles found. Try broader skills.",
        }

    system_prompt = (
        "You are a senior tech recruiter evaluating REAL candidates from LinkedIn, "
        "Google Scholar, and GitHub open-source contributor profiles.\n"
        "Score each candidate 1-10 based on: skill match, experience level match, "
        "location match, and profile quality.\n"
        "Be strict — reserve 8-10 for outstanding matches, 6-7 for good matches, "
        "1-4 for poor matches. Only score 5 if you are genuinely unsure.\n\n"
        "Return ONLY a JSON array with:\n"
        '- "index": The candidate\'s _batch_idx value\n'
        '- "score": 1-10\n'
        '- "reason": One sentence explaining the score\n'
        "Return ONLY the JSON array."
    )

    # Score in batches; use _batch_idx so scores map back to global positions
    eval_map: dict[int, dict] = {}
    for batch_start in range(0, len(candidates), ENRICH_BATCH_SIZE):
        batch = candidates[batch_start : batch_start + ENRICH_BATCH_SIZE]
        indexed_batch = [{"_batch_idx": i, **c} for i, c in enumerate(batch)]
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Required: skills={filters.get('skills')}, "
                f"level={filters.get('experience_level', 'any')}, "
                f"location={filters.get('location', 'any')}\n\n"
                f"Candidates:\n{json.dumps(indexed_batch, indent=2)}"
            )),
        ]
        response = llm.invoke(messages)
        evaluations = _extract_json(response.content)
        if not isinstance(evaluations, list):
            evaluations = []
        for ev in evaluations:
            if isinstance(ev, dict) and "index" in ev:
                global_idx = batch_start + ev["index"]
                eval_map[global_idx] = ev

    evaluated = []
    for i, c in enumerate(candidates):
        ev = eval_map.get(i, {"score": 5, "reason": "Not evaluated"})
        c["_score"] = ev.get("score", 5)
        c["_reason"] = ev.get("reason", "")
        evaluated.append(c)

    evaluated.sort(key=lambda x: x.get("_score", 0), reverse=True)

    top_count = sum(1 for c in evaluated if c.get("_score", 0) >= MIN_CANDIDATE_SCORE)
    feedback = ""
    if top_count < 3:
        feedback = "Too few qualified results. Broaden the query — fewer skills, drop location."

    return {
        "evaluated_candidates": evaluated,
        "refinement_feedback": feedback,
    }


def decide(state: AgentState) -> Literal["refine", "finish"]:
    feedback = state.get("refinement_feedback", "")
    count = state.get("refinement_count", 0)
    if feedback and count < MAX_REFINE_LOOPS:
        return "refine"
    return "finish"


def prepare_refinement(state: AgentState) -> dict:
    return {"refinement_count": state.get("refinement_count", 0) + 1}


def finalize(state: AgentState) -> dict:
    """Return only the top-scoring qualified candidates, capped at TOP_CANDIDATES_LIMIT."""
    evaluated = state.get("evaluated_candidates", [])
    # Already sorted descending by score; filter out low scorers first
    top = [c for c in evaluated if c.get("_score", 0) >= MIN_CANDIDATE_SCORE]
    top = top[:TOP_CANDIDATES_LIMIT]
    print(
        f"[Agent] Finalizing: {len(evaluated)} evaluated → "
        f"{len(top)} top candidates (score ≥ {MIN_CANDIDATE_SCORE}, cap {TOP_CANDIDATES_LIMIT})"
    )
    return {"final_candidates": top}


# ══════════════════════════════════════════════════════════════════════
# Build the LangGraph
# ══════════════════════════════════════════════════════════════════════
def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("plan_search", plan_search)
    graph.add_node("search_sources", search_sources)
    graph.add_node("enrich", enrich_candidates)
    graph.add_node("evaluate", evaluate_candidates)
    graph.add_node("refine", prepare_refinement)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("plan_search")
    graph.add_edge("plan_search", "search_sources")
    graph.add_edge("search_sources", "enrich")
    graph.add_edge("enrich", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        decide,
        {"refine": "refine", "finish": "finalize"},
    )

    graph.add_edge("refine", "plan_search")
    graph.add_edge("finalize", END)

    return graph


agent_graph = _build_graph().compile()


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════
def _to_candidate(data: dict, search_skills: list[str]) -> Candidate:
    name = data.get("name", "Unknown")
    cid = hashlib.md5(f"{name}-{data.get('headline', '')}".encode()).hexdigest()[:12]

    profile_url = data.get("profile_url", "")
    if not profile_url:
        slug = re.sub(r"[^a-z0-9-]", "", name.lower().replace(" ", "-"))
        profile_url = f"https://duckduckgo.com/?q={slug}"

    cand_skills = data.get("matched_skills", [])
    if not cand_skills:
        text = f"{data.get('headline', '')} {data.get('snippet', '')}".lower()
        cand_skills = [s for s in search_skills if s.lower() in text]

    return Candidate(
        id=cid,
        name=name,
        headline=data.get("headline", ""),
        location=data.get("location", ""),
        profile_url=profile_url,
        snippet=data.get("snippet", ""),
        matched_skills=cand_skills[:6],
        avatar_initials=_get_initials(name),
    )


async def search_linkedin_profiles(
    filters: SearchFilters,
) -> tuple[list[Candidate], bool]:
    """Run the LangGraph agent to discover candidates across multiple sources."""
    cache_key = f"{sorted(filters.skills)}:{filters.experience_level}:{filters.location}:{filters.page}"
    key = hashlib.md5(cache_key.encode()).hexdigest()

    if key in _cache:
        ts, cached = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return cached, False

    initial_state: AgentState = {
        "filters": {
            "skills": filters.skills,
            "experience_level": filters.experience_level or "",
            "location": filters.location or "",
        },
        "search_query": "",
        "raw_results": [],
        "parsed_candidates": [],
        "enriched_candidates": [],
        "evaluated_candidates": [],
        "refinement_feedback": "",
        "refinement_count": 0,
        "final_candidates": [],
    }

    try:
        print("[Agent] Starting LangGraph candidate search...")
        result = agent_graph.invoke(initial_state)
        raw_list = result.get("final_candidates", [])

        candidates = []
        seen_urls = set()
        for data in raw_list:
            try:
                candidate = _to_candidate(data, filters.skills)
                dedupe_key = _canonicalize_url(candidate.profile_url)
                if dedupe_key and dedupe_key in seen_urls:
                    continue
                if dedupe_key:
                    seen_urls.add(dedupe_key)
                candidates.append(candidate)
            except Exception:
                continue

        print(f"[Agent] Returning {len(candidates)} real candidates")

        if candidates:
            _cache[key] = (time.time(), candidates)

        return candidates, False

    except Exception as e:
        print(f"[LangGraph Agent Error] {e}")
        import traceback
        traceback.print_exc()
        return [], False

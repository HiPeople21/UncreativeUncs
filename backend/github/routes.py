"""FastAPI routes for GitHub Tools integration."""

from typing import Optional, List
from fastapi import APIRouter, Query

from .tools import get_github_tools
from .code_quality_agent import get_code_quality_agent
from models import (
    RepositorySearchResponse,
    ContributorsResponse,
    ContributorCommitsResponse,
    CodeQualityResponse,
)

# Create router for GitHub endpoints
router = APIRouter(prefix="/api/github", tags=["github"])

# ============================================================================
# Repository Search Routes
# ============================================================================

@router.get("/search/repos", response_model=RepositorySearchResponse)
async def search_repositories(
    tag: str = Query(..., description="Topic/tag to search for"),
    language: Optional[str] = Query(None, description="Programming language filter"),
    min_stars: int = Query(0, ge=0, description="Minimum star count"),
    sort: str = Query("stars", description="Sort by 'stars', 'forks', 'updated', or 'score'"),
    order: str = Query("desc", description="'asc' or 'desc'"),
    max_results: int = Query(30, ge=1, le=100, description="Maximum results to return"),
):
    """
    Search for GitHub repositories by topic/tag.
    
    Args:
        tag: Topic or tag to search for (required)
        language: Optional programming language filter
        min_stars: Minimum star count filter
        sort: Sort criteria (stars, forks, updated, score)
        order: Ascending or descending
        max_results: Number of results to return (max 100)
    
    Returns:
        Repository search results with metadata
    """
    github_tools = get_github_tools()
    return github_tools.search_repos(
        tag=tag,
        language=language,
        min_stars=min_stars,
        sort=sort,
        order=order,
        max_results=max_results
    )


@router.get("/search/repos/batch", response_model=RepositorySearchResponse)
async def search_multiple_repositories(
    tags: List[str] = Query(..., description="List of tags to search"),
    language: Optional[str] = Query(None, description="Programming language filter"),
    min_stars: int = Query(0, ge=0, description="Minimum star count"),
    max_results: int = Query(30, ge=1, le=100, description="Maximum results to return"),
):
    """
    Search for repositories with all specified tags.
    
    Args:
        tags: List of tags to search for (repos must have ALL tags)
        language: Optional programming language filter
        min_stars: Minimum star count filter
        max_results: Number of results to return (max 100)
    
    Returns:
        Repositories that have all specified tags
    """
    github_tools = get_github_tools()
    return github_tools.search_multiple_repos(
        tags=tags,
        language=language,
        min_stars=min_stars,
        max_results=max_results
    )


# ============================================================================
# Repository Contributors Routes
# ============================================================================

@router.get("/repos/{owner}/{repo}/contributors", response_model=ContributorsResponse)
async def get_repository_contributors(
    owner: str,
    repo: str,
    max_results: int = Query(50, ge=1, le=100, description="Maximum contributors to fetch"),
):
    """
    Get top contributors for a repository.
    
    Args:
        owner: Repository owner/organization name
        repo: Repository name
        max_results: Maximum number of contributors to return
    
    Returns:
        List of contributors with contribution counts
    """
    github_tools = get_github_tools()
    return github_tools.get_contributors(
        owner=owner,
        repo=repo,
        max_results=max_results
    )


# ============================================================================
# Contributor Commits Routes
# ============================================================================

@router.get(
    "/repos/{owner}/{repo}/contributors/{contributor}/commits",
    response_model=ContributorCommitsResponse
)
async def get_contributor_commits(
    owner: str,
    repo: str,
    contributor: str,
    max_results: int = Query(50, ge=1, le=100, description="Maximum commits to fetch"),
):
    """
    Get commits by a specific contributor in a repository.
    
    Args:
        owner: Repository owner/organization name
        repo: Repository name
        contributor: Contributor login name
        max_results: Maximum number of commits to return
    
    Returns:
        List of commits by the specified contributor
    """
    github_tools = get_github_tools()
    return github_tools.get_commits(
        owner=owner,
        repo=repo,
        contributor=contributor,
        max_results=max_results
    )


@router.get("/repos/{owner}/{repo}/contributors/commits/all")
async def get_all_contributors_commits(
    owner: str,
    repo: str,
    max_contributors: int = Query(
        10, ge=1, le=50, description="Maximum number of top contributors"
    ),
    commits_per_contributor: int = Query(
        20, ge=1, le=100, description="Commits per contributor"
    ),
):
    """
    Get recent commits from all top contributors in a repository.
    
    Args:
        owner: Repository owner/organization name
        repo: Repository name
        max_contributors: Number of top contributors to fetch
        commits_per_contributor: Commits to fetch per contributor
    
    Returns:
        Dictionary mapping contributor login to their commits
    """
    github_tools = get_github_tools()
    result = github_tools.analyzer.get_all_contributors_commits(
        owner=owner,
        repo=repo,
        max_contributors=max_contributors,
        commits_per_contributor=commits_per_contributor
    )
    return result


# ============================================================================
# Code Quality Analysis Routes
# ============================================================================

@router.get("/repos/{owner}/{repo}/contributors/{contributor}/quality", response_model=CodeQualityResponse)
async def analyze_contributor_code_quality(
    owner: str,
    repo: str,
    contributor: str,
    max_results: int = Query(10, ge=1, le=50, description="Maximum commits to grade"),
):
    """
    Grade code quality of a contributor's commits using AI.
    
    Provides objective quality scores (0-100) across multiple dimensions:
    - Maintainability: Code structure and modularity
    - Readability: Code clarity and documentation
    - Test Coverage: Testing practices
    - Complexity: Code simplicity (higher = simpler)
    - Performance: Performance characteristics
    - Security: Security practices
    
    Args:
        owner: Repository owner/organization name
        repo: Repository name
        contributor: Contributor login name
        max_results: Maximum number of commits to grade
    
    Returns:
        Quality grades for each commit
    """
    print(f"\n[API] Code quality grading requested for {owner}/{repo} contributor: {contributor}")
    print(f"[API] Max commits to grade: {max_results}")
    
    try:
        print(f"[API] Initializing GitHub tools...")
        github_tools = get_github_tools()
        print(f"[API] Initializing code quality agent...")
        agent = get_code_quality_agent()
        
        # Get commits for the contributor
        print(f"[API] Fetching commits for contributor {contributor}...")
        commits_response = github_tools.get_commits(
            owner=owner,
            repo=repo,
            contributor=contributor,
            max_results=max_results
        )
        print(f"[API] Fetched {commits_response.total_commits} commits")
        
        if not commits_response.success:
            print(f"[API] ERROR: Failed to fetch commits - {commits_response.error}")
            return CodeQualityResponse(
                owner=owner,
                repo=repo,
                contributor=contributor,
                metrics=[],
                total_analyzed=0,
                success=False,
                error=commits_response.error
            )
        
        # Analyze each commit's quality
        print(f"[API] Preparing {len(commits_response.commits)} commits for grading...")
        commits_data = [
            {
                "sha": commit.sha,
                "diff": commit.diff or "",
                "message": commit.message
            }
            for commit in commits_response.commits
        ]
        
        print(f"[API] Starting AI grading...")
        metrics = await agent.analyze_commits_batch(commits_data)
        print(f"[API] Grading complete! Generated {len(metrics)} quality reports")
        
        return CodeQualityResponse(
            owner=owner,
            repo=repo,
            contributor=contributor,
            metrics=metrics,
            total_analyzed=len(metrics),
            success=True
        )
    
    except Exception as e:
        print(f"[API] ERROR: Exception during code quality grading - {str(e)}")
        import traceback
        traceback.print_exc()
        return CodeQualityResponse(
            owner=owner,
            repo=repo,
            contributor=contributor,
            metrics=[],
            total_analyzed=0,
            success=False,
            error=str(e)
        )


# ============================================================================
# Health Check Route
# ============================================================================

@router.get("/health")
async def github_api_health():
    """Health check for GitHub API routes."""
    return {
        "status": "ok",
        "service": "github_tools",
        "endpoints": [
            "GET /api/github/search/repos",
            "GET /api/github/search/repos/batch",
            "GET /api/github/repos/{owner}/{repo}/contributors",
            "GET /api/github/repos/{owner}/{repo}/contributors/{contributor}/commits",
            "GET /api/github/repos/{owner}/{repo}/contributors/commits/all",
            "GET /api/github/repos/{owner}/{repo}/contributors/{contributor}/quality",
            "GET /api/github/health",
        ]
    }

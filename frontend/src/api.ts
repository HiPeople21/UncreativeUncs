/** API client for the FastAPI backend. */

import type { SearchFilters, SearchResponse } from "./types";

const API_BASE = "http://localhost:8000";

export async function searchCandidates(
    filters: SearchFilters,
    page: number = 1
): Promise<SearchResponse> {
    const params = new URLSearchParams();

    if (filters.skills.length > 0) {
        params.set("skills", filters.skills.join(","));
    }
    if (filters.experience_level) {
        params.set("experience_level", filters.experience_level);
    }
    if (filters.location) {
        params.set("location", filters.location);
    }
    params.set("page", String(page));

    const resp = await fetch(`${API_BASE}/api/candidates?${params.toString()}`);
    if (!resp.ok) {
        throw new Error(`API error: ${resp.status}`);
    }
    return resp.json();
}

export async function healthCheck(): Promise<boolean> {
    try {
        const resp = await fetch(`${API_BASE}/api/health`);
        return resp.ok;
    } catch {
        return false;
    }
}

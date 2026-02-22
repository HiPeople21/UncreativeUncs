/** API client for the FastAPI backend. */

import type { Candidate, SearchFilters, SearchResponse } from "./types";

const API_BASE = "http://localhost:8000";

export interface AgentStep {
    type: "start" | "thinking" | "searching" | "success" | "refining" | "error" | "done";
    message: string;
    detail: string;
}

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
    if (filters.description) {
        params.set("description", filters.description);
    }
    params.set("page", String(page));

    const resp = await fetch(`${API_BASE}/api/pipeline?${params.toString()}`);
    if (!resp.ok) {
        throw new Error(`API error: ${resp.status}`);
    }

    const data = await resp.json();

    // Map pipeline response to SearchResponse shape
    const candidates: Candidate[] = (data.candidates ?? []).map((c: any) => {
        const bestUrl =
            c.profile_urls?.linkedin ||
            c.profile_urls?.github ||
            Object.values(c.profile_urls ?? {})[0] ||
            "";
        const nameParts = (c.name ?? "Unknown").split(" ");
        const initials = nameParts
            .slice(0, 2)
            .map((p: string) => p[0]?.toUpperCase() ?? "")
            .join("");

        return {
            id: c.canonical_id ?? "",
            name: c.name ?? "Unknown",
            headline: c.headline ?? "",
            location: c.location ?? "",
            profile_url: bestUrl,
            snippet: c.final_score != null ? `Score: ${c.final_score}` : "",
            matched_skills: c.skills ?? [],
            avatar_initials: initials,
        } satisfies Candidate;
    });

    return {
        candidates,
        total_results: data.total ?? candidates.length,
        page,
        has_more: false,
    };
}

export function subscribeAgentStatus(
    onStep: (step: AgentStep) => void,
    onDone: () => void
): () => void {
    const evtSource = new EventSource(`${API_BASE}/api/agent-status`);

    evtSource.onmessage = (event) => {
        try {
            const step: AgentStep = JSON.parse(event.data);
            if (step.type === "done") {
                onDone();
                evtSource.close();
            } else {
                onStep(step);
            }
        } catch {
            // ignore parse errors
        }
    };

    evtSource.onerror = () => {
        evtSource.close();
        onDone();
    };

    return () => evtSource.close();
}

export async function healthCheck(): Promise<boolean> {
    try {
        const resp = await fetch(`${API_BASE}/api/health`);
        return resp.ok;
    } catch {
        return false;
    }
}

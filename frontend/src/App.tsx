import { useState } from "react";
import FilterSidebar from "./components/FilterSidebar";
import CandidateGrid from "./components/CandidateGrid";
import CandidateModal from "./components/CandidateModal";
import { searchCandidates } from "./api";
import type { Candidate, SearchFilters } from "./types";

export default function App() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeFilters, setActiveFilters] = useState<SearchFilters | null>(null);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(false);

  async function handleSearch(filters: SearchFilters) {
    setIsLoading(true);
    setError(null);
    setActiveFilters(filters);
    setPage(1);

    try {
      const data = await searchCandidates(filters, 1);
      setCandidates(data.candidates);
      setHasMore(data.has_more);
      setHasSearched(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      setCandidates([]);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleLoadMore() {
    if (!activeFilters || isLoading) return;
    const nextPage = page + 1;
    setIsLoading(true);

    try {
      const data = await searchCandidates(activeFilters, nextPage);
      setCandidates((prev) => [...prev, ...data.candidates]);
      setHasMore(data.has_more);
      setPage(nextPage);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load more");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                <path d="M16 3.13a4 4 0 0 1 0 7.75" />
              </svg>
            </div>
            <div>
              <h1>TalentScope</h1>
              <span className="tagline">LinkedIn Candidate Discovery</span>
            </div>
          </div>
          {hasSearched && (
            <div className="result-count">
              <span className="count-number">{candidates.length}</span>
              <span className="count-label">candidates found</span>
            </div>
          )}
        </div>
      </header>

      <main className="app-main">
        <FilterSidebar onSearch={handleSearch} isLoading={isLoading} />

        <section className="results-section">
          {error && (
            <div className="error-banner">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
              {error}
            </div>
          )}

          {activeFilters && hasSearched && (
            <div className="active-filters-bar">
              {activeFilters.skills.map((skill) => (
                <span key={skill} className="active-filter-pill">
                  {skill}
                </span>
              ))}
              {activeFilters.experience_level && (
                <span className="active-filter-pill">
                  {activeFilters.experience_level}
                </span>
              )}
              {activeFilters.location && (
                <span className="active-filter-pill">
                  📍 {activeFilters.location}
                </span>
              )}
            </div>
          )}

          <CandidateGrid
            candidates={candidates}
            isLoading={isLoading && !hasSearched}
            hasSearched={hasSearched}
            onCandidateClick={setSelectedCandidate}
          />

          {hasMore && !isLoading && (
            <div className="load-more-container">
              <button className="load-more-btn" onClick={handleLoadMore}>
                Load More Candidates
              </button>
            </div>
          )}

          {isLoading && hasSearched && (
            <div className="loading-more">
              <span className="spinner" /> AI agent is finding more candidates...
            </div>
          )}
        </section>
      </main>

      {selectedCandidate && (
        <CandidateModal
          candidate={selectedCandidate}
          onClose={() => setSelectedCandidate(null)}
        />
      )}
    </div>
  );
}

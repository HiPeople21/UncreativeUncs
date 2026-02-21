import type { Candidate } from "../types";
import CandidateCard from "./CandidateCard";

interface Props {
    candidates: Candidate[];
    isLoading: boolean;
    hasSearched: boolean;
    onCandidateClick: (candidate: Candidate) => void;
}

export default function CandidateGrid({
    candidates,
    isLoading,
    hasSearched,
    onCandidateClick,
}: Props) {
    if (isLoading) {
        return (
            <div className="candidate-grid">
                {Array.from({ length: 6 }).map((_, i) => (
                    <div key={i} className="candidate-card skeleton">
                        <div className="skeleton-header">
                            <div className="skeleton-avatar" />
                            <div className="skeleton-text">
                                <div className="skeleton-line wide" />
                                <div className="skeleton-line narrow" />
                            </div>
                        </div>
                        <div className="skeleton-line medium" />
                        <div className="skeleton-chips">
                            <div className="skeleton-chip" />
                            <div className="skeleton-chip" />
                            <div className="skeleton-chip" />
                        </div>
                        <div className="skeleton-line wide" />
                        <div className="skeleton-line medium" />
                    </div>
                ))}
            </div>
        );
    }

    if (!hasSearched) {
        return (
            <div className="empty-state">
                <div className="empty-icon">
                    <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="11" cy="11" r="8" />
                        <line x1="21" y1="21" x2="16.65" y2="16.65" />
                    </svg>
                </div>
                <h2>Find Your Next Great Hire</h2>
                <p>Select technical skills and filters, then search to discover candidates on LinkedIn.</p>
            </div>
        );
    }

    if (candidates.length === 0) {
        return (
            <div className="empty-state">
                <div className="empty-icon">
                    <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                        <circle cx="9" cy="7" r="4" />
                        <line x1="23" y1="11" x2="17" y2="11" />
                    </svg>
                </div>
                <h2>No Candidates Found</h2>
                <p>
                    Try adjusting your filters or broadening your search criteria.
                    Google may also rate-limit requests — try again in a moment.
                </p>
            </div>
        );
    }

    return (
        <div className="candidate-grid">
            {candidates.map((candidate) => (
                <CandidateCard
                    key={candidate.id}
                    candidate={candidate}
                    onClick={() => onCandidateClick(candidate)}
                />
            ))}
        </div>
    );
}

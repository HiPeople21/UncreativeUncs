import type { Candidate } from "../types";

interface Props {
    candidate: Candidate;
    onClick: () => void;
}

export default function CandidateCard({ candidate, onClick }: Props) {
    // Generate a deterministic gradient based on the candidate's name
    const hue = candidate.name
        .split("")
        .reduce((acc, c) => acc + c.charCodeAt(0), 0) % 360;

    return (
        <div className="candidate-card" onClick={onClick}>
            <div className="card-header">
                <div
                    className="avatar"
                    style={{
                        background: `linear-gradient(135deg, hsl(${hue}, 70%, 50%), hsl(${(hue + 60) % 360}, 70%, 40%))`,
                    }}
                >
                    {candidate.avatar_initials}
                </div>
                <div className="card-info">
                    <h3 className="candidate-name">{candidate.name}</h3>
                    <p className="candidate-headline">{candidate.headline}</p>
                </div>
            </div>

            {candidate.location && (
                <div className="candidate-location">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
                        <circle cx="12" cy="10" r="3" />
                    </svg>
                    {candidate.location}
                </div>
            )}

            {candidate.matched_skills.length > 0 && (
                <div className="candidate-skills">
                    {candidate.matched_skills.map((skill) => (
                        <span key={skill} className="skill-tag">
                            {skill}
                        </span>
                    ))}
                </div>
            )}

            {candidate.snippet && (
                <p className="candidate-snippet">{candidate.snippet.slice(0, 150)}...</p>
            )}

            <div className="card-footer">
                <a
                    className="linkedin-link"
                    href={candidate.profile_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                    </svg>
                    View Profile
                </a>
                <button className="details-btn" onClick={onClick}>
                    Details →
                </button>
            </div>
        </div>
    );
}

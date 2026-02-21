import { useState, type FormEvent } from "react";
import type { SearchFilters } from "../types";

const POPULAR_SKILLS = [
    "Python", "JavaScript", "TypeScript", "React", "Node.js",
    "Java", "Go", "Rust", "C++", "C#",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes",
    "SQL", "PostgreSQL", "MongoDB", "Redis", "GraphQL",
    "Machine Learning", "Data Science", "DevOps", "CI/CD",
    "Swift", "Kotlin", "Flutter", "Vue.js", "Angular",
    "Terraform", "Linux", "Git", "Agile", "Scrum",
];

const EXPERIENCE_LEVELS = [
    { value: "", label: "Any Level" },
    { value: "junior", label: "Junior / Entry" },
    { value: "mid", label: "Mid-Level" },
    { value: "senior", label: "Senior" },
    { value: "lead", label: "Lead / Manager" },
    { value: "principal", label: "Principal / Staff" },
];

const LOCATIONS = [
    { value: "", label: "Any Location" },
    { value: "Remote", label: "🌍 Remote" },
    { value: "San Francisco", label: "🇺🇸 San Francisco" },
    { value: "New York", label: "🇺🇸 New York" },
    { value: "Seattle", label: "🇺🇸 Seattle" },
    { value: "Austin", label: "🇺🇸 Austin" },
    { value: "London", label: "🇬🇧 London" },
    { value: "Berlin", label: "🇩🇪 Berlin" },
    { value: "Toronto", label: "🇨🇦 Toronto" },
    { value: "Singapore", label: "🇸🇬 Singapore" },
    { value: "Sydney", label: "🇦🇺 Sydney" },
    { value: "Bangalore", label: "🇮🇳 Bangalore" },
];

interface Props {
    onSearch: (filters: SearchFilters) => void;
    isLoading: boolean;
}

export default function FilterSidebar({ onSearch, isLoading }: Props) {
    const [selectedSkills, setSelectedSkills] = useState<string[]>([]);
    const [customSkill, setCustomSkill] = useState("");
    const [experienceLevel, setExperienceLevel] = useState("");
    const [location, setLocation] = useState("");
    const [showAllSkills, setShowAllSkills] = useState(false);

    const displayedSkills = showAllSkills ? POPULAR_SKILLS : POPULAR_SKILLS.slice(0, 15);

    function toggleSkill(skill: string) {
        setSelectedSkills((prev) =>
            prev.includes(skill) ? prev.filter((s) => s !== skill) : [...prev, skill]
        );
    }

    function addCustomSkill(e: FormEvent) {
        e.preventDefault();
        const trimmed = customSkill.trim();
        if (trimmed && !selectedSkills.includes(trimmed)) {
            setSelectedSkills((prev) => [...prev, trimmed]);
            setCustomSkill("");
        }
    }

    function handleSearch() {
        onSearch({
            skills: selectedSkills,
            experience_level: experienceLevel,
            location,
        });
    }

    function handleClear() {
        setSelectedSkills([]);
        setExperienceLevel("");
        setLocation("");
    }

    const hasActiveFilters = selectedSkills.length > 0 || experienceLevel || location;

    return (
        <aside className="filter-sidebar">
            <div className="filter-header">
                <h2>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
                    </svg>
                    Filters
                </h2>
                {hasActiveFilters && (
                    <button className="clear-btn" onClick={handleClear}>
                        Clear All
                    </button>
                )}
            </div>

            {/* Skills */}
            <div className="filter-section">
                <h3>Technical Skills</h3>
                <div className="skill-chips">
                    {displayedSkills.map((skill) => (
                        <button
                            key={skill}
                            className={`chip ${selectedSkills.includes(skill) ? "active" : ""}`}
                            onClick={() => toggleSkill(skill)}
                        >
                            {skill}
                            {selectedSkills.includes(skill) && <span className="chip-x">×</span>}
                        </button>
                    ))}
                </div>
                {!showAllSkills && (
                    <button className="show-more-btn" onClick={() => setShowAllSkills(true)}>
                        Show all {POPULAR_SKILLS.length} skills ↓
                    </button>
                )}
                <form className="custom-skill-form" onSubmit={addCustomSkill}>
                    <input
                        type="text"
                        placeholder="Add a custom skill..."
                        value={customSkill}
                        onChange={(e) => setCustomSkill(e.target.value)}
                        className="custom-skill-input"
                    />
                    <button type="submit" className="add-skill-btn" disabled={!customSkill.trim()}>
                        +
                    </button>
                </form>
                {/* Show custom skills added */}
                {selectedSkills.filter((s) => !POPULAR_SKILLS.includes(s)).length > 0 && (
                    <div className="custom-skills-list">
                        {selectedSkills
                            .filter((s) => !POPULAR_SKILLS.includes(s))
                            .map((skill) => (
                                <button
                                    key={skill}
                                    className="chip active custom"
                                    onClick={() => toggleSkill(skill)}
                                >
                                    {skill} <span className="chip-x">×</span>
                                </button>
                            ))}
                    </div>
                )}
            </div>

            {/* Experience Level */}
            <div className="filter-section">
                <h3>Experience Level</h3>
                <div className="radio-group">
                    {EXPERIENCE_LEVELS.map((level) => (
                        <label
                            key={level.value}
                            className={`radio-option ${experienceLevel === level.value ? "selected" : ""}`}
                        >
                            <input
                                type="radio"
                                name="experience"
                                value={level.value}
                                checked={experienceLevel === level.value}
                                onChange={(e) => setExperienceLevel(e.target.value)}
                            />
                            <span className="radio-dot" />
                            {level.label}
                        </label>
                    ))}
                </div>
            </div>

            {/* Location */}
            <div className="filter-section">
                <h3>Location</h3>
                <select
                    className="location-select"
                    value={location}
                    onChange={(e) => setLocation(e.target.value)}
                >
                    {LOCATIONS.map((loc) => (
                        <option key={loc.value} value={loc.value}>
                            {loc.label}
                        </option>
                    ))}
                </select>
            </div>

            {/* Search Button */}
            <button
                className="search-btn"
                onClick={handleSearch}
                disabled={isLoading || !hasActiveFilters}
            >
                {isLoading ? (
                    <>
                        <span className="spinner" /> AI Agent is searching...
                    </>
                ) : (
                    <>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="11" cy="11" r="8" />
                            <line x1="21" y1="21" x2="16.65" y2="16.65" />
                        </svg>
                        Search Candidates
                    </>
                )}
            </button>
        </aside>
    );
}

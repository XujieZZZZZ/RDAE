"""
Experience Tracker for cross-run learning.
Extracts successful bug fixes from bug_history and consolidates them into reusable experiences.
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from llm import call_llm

logger = logging.getLogger("aide")


@dataclass
class Experience:
    """Single experience record from successful bug fixes"""
    experience_id: str
    category: str  # e.g., "API_ERROR", "LOGIC_ERROR", "TYPE_ERROR"
    pattern: str  # General pattern description
    solution: str  # General solution approach
    source_bugs: List[str]  # List of bug_ids that contributed to this experience
    confidence: float  # 0-1, based on number of successful applications
    created_at: float
    last_updated: float
    application_count: int = 0  # How many times this experience was successfully applied

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Experience':
        return cls(**data)


class ExperienceTracker:
    """Tracks and manages cross-run experiences"""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        # Experience directory is at the same level as workspaces directory
        # workspace_dir is like: /path/to/workspaces/0-carbon_aware_optimization
        # we want: /path/to/experiences
        workspaces_parent = self.workspace_dir.parent.parent  # Go up two levels from workspace
        self.experience_dir = workspaces_parent / "experiences"
        self.experience_file = self.experience_dir / "bug_experiences.json"
        self.experiences: List[Experience] = []
        self._ensure_experience_dir()
        self._load_experiences()

    def _ensure_experience_dir(self) -> None:
        """Ensure experience directory exists"""
        self.experience_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Experience directory: {self.experience_dir}")

    def _load_experiences(self) -> None:
        """Load existing experiences from file"""
        if not self.experience_file.exists():
            logger.info("No existing experience file, starting fresh")
            return

        try:
            with open(self.experience_file, 'r') as f:
                data = json.load(f)
                self.experiences = [Experience.from_dict(exp) for exp in data.get("experiences", [])]
            logger.info(f"Loaded {len(self.experiences)} experiences from {self.experience_file}")
        except Exception as e:
            logger.error(f"Failed to load experiences: {e}")
            self.experiences = []

    def _save_experiences(self) -> None:
        """Save experiences to file"""
        try:
            data = {
                "experiences": [exp.to_dict() for exp in self.experiences],
                "metadata": {
                    "total_experiences": len(self.experiences),
                    "last_updated": time.time()
                }
            }

            with open(self.experience_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.experiences)} experiences to {self.experience_file}")
        except Exception as e:
            logger.error(f"Failed to save experiences: {e}")

    def extract_and_consolidate_experiences(self, bug_history_file: Path) -> None:
        """
        Extract successful bug fixes from bug_history and consolidate into experiences.

        Args:
            bug_history_file: Path to bug_history.json
        """
        if not bug_history_file.exists():
            logger.info("No bug history file found, skipping experience extraction")
            return

        try:
            with open(bug_history_file, 'r') as f:
                bug_data = json.load(f)

            bugs = bug_data.get("bugs", [])

            # Filter bugs with successful fixes
            resolved_bugs = [bug for bug in bugs if bug.get("successful_fixes")]

            if not resolved_bugs:
                logger.info("No resolved bugs found in this run")
                return

            logger.info(f"Found {len(resolved_bugs)} resolved bugs, consolidating into experiences...")

            # Use LLM to consolidate experiences

            # Prepare bug information for LLM
            bug_summaries = []
            for bug in resolved_bugs:
                bug_summaries.append({
                    "bug_id": bug.get("bug_id"),
                    "error_type": bug.get("error_type"),
                    "error_reason": bug.get("error_reason"),
                    "fix_suggestion": bug.get("fix_suggestion"),
                    "summary": bug.get("summary"),
                    "successful_fix": bug.get("successful_fixes", [])[-1] if bug.get("successful_fixes") else None,
                    "occurrence_count": bug.get("occurrence_count", 1)
                })

            bug_summaries_json = json.dumps(bug_summaries, indent=2)

            # Get existing experiences for context
            existing_experiences_json = json.dumps(
                [exp.to_dict() for exp in self.experiences],
                indent=2
            ) if self.experiences else "[]"

            prompt = f"""
You are an expert in extracting reusable programming experiences from bug fix histories.

【TASK】
Analyze the resolved bugs below and consolidate them into general, reusable experiences.
Each experience should capture a PATTERN and SOLUTION that can be applied to future similar problems.

【RESOLVED BUGS FROM THIS RUN】
{bug_summaries_json}

【EXISTING EXPERIENCES (from previous runs)】
{existing_experiences_json}

【CONSOLIDATION GUIDELINES】

1. **Generalize**: Extract the underlying pattern, not specific variable/class names
2. **Categorize**: Assign to one of these categories:
   - API_ERROR: Misunderstanding of framework/library API
   - LOGIC_ERROR: Incorrect computation or algorithm logic
   - TYPE_ERROR: Data structure type confusion
   - CONSTRAINT_ERROR: Violation of constraints or invariants
   - OTHER: Other types of errors

3. **Merge similar experiences**: If a new bug is similar to an existing experience, UPDATE that experience instead of creating a new one

4. **Solution format**: Describe the general approach, not specific code

【OUTPUT FORMAT】
Respond with ONLY valid JSON (no markdown, no code blocks):

{{
  "new_experiences": [
    {{
      "category": "API_ERROR|LOGIC_ERROR|TYPE_ERROR|CONSTRAINT_ERROR|OTHER",
      "pattern": "General description of the error pattern",
      "solution": "General solution approach that worked",
      "source_bugs": ["bug_id1", "bug_id2"],
      "confidence": 0.8
    }}
  ],
  "updated_experiences": [
    {{
      "experience_id": "existing_experience_id",
      "updated_pattern": "Updated pattern description",
      "updated_solution": "Updated solution description",
      "additional_source_bugs": ["bug_id3"],
      "confidence_adjustment": 0.1
    }}
  ]
}}

CRITICAL RULES:
1. Output ONLY JSON, no markdown
2. Generalize patterns - remove specific names
3. Solutions should be actionable but general
4. Merge similar patterns into existing experiences when possible
5. Confidence should reflect how reliable the solution is (0.5-1.0)
"""

            response = call_llm(prompt)

            # Parse LLM response
            response = response.strip()
            if response.startswith("```"):
                lines = response.split('\n')
                start_idx = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith("{"):
                        start_idx = i
                        break
                end_idx = len(lines)
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().endswith("}"):
                        end_idx = i + 1
                        break
                response = '\n'.join(lines[start_idx:end_idx])

            result = json.loads(response)

            # Process new experiences
            new_count = 0
            for new_exp_data in result.get("new_experiences", []):
                import uuid
                experience = Experience(
                    experience_id=f"exp_{uuid.uuid4().hex[:8]}",
                    category=new_exp_data.get("category", "OTHER"),
                    pattern=new_exp_data.get("pattern", ""),
                    solution=new_exp_data.get("solution", ""),
                    source_bugs=new_exp_data.get("source_bugs", []),
                    confidence=new_exp_data.get("confidence", 0.5),
                    created_at=time.time(),
                    last_updated=time.time(),
                    application_count=0
                )
                self.experiences.append(experience)
                new_count += 1
                logger.info(f"Created new experience: {experience.experience_id} ({experience.category})")

            # Process updated experiences
            update_count = 0
            for update_data in result.get("updated_experiences", []):
                exp_id = update_data.get("experience_id")
                exp = next((e for e in self.experiences if e.experience_id == exp_id), None)
                if exp:
                    exp.pattern = update_data.get("updated_pattern", exp.pattern)
                    exp.solution = update_data.get("updated_solution", exp.solution)
                    exp.source_bugs.extend(update_data.get("additional_source_bugs", []))
                    exp.confidence = min(1.0, exp.confidence + update_data.get("confidence_adjustment", 0))
                    exp.last_updated = time.time()
                    update_count += 1
                    logger.info(f"Updated experience: {exp_id}")

            # Save experiences
            self._save_experiences()
            logger.info(f"Experience consolidation complete: {new_count} new, {update_count} updated")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
        except Exception as e:
            logger.error(f"Failed to extract experiences: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def get_experience_summary(self, max_experiences: int = 10) -> str:
        """Get formatted experience summary for LLM"""
        if not self.experiences:
            return ""

        # Sort by confidence and application count
        sorted_experiences = sorted(
            self.experiences,
            key=lambda e: (e.confidence, e.application_count),
            reverse=True
        )[:max_experiences]

        summary_parts = ["【LEARNED EXPERIENCES FROM PREVIOUS RUNS】"]
        summary_parts.append("These are proven solutions from past successful bug fixes.\n")

        for i, exp in enumerate(sorted_experiences, 1):
            summary_parts.append(f"Experience #{i} [{exp.category}] (confidence: {exp.confidence:.2f})")
            summary_parts.append(f"PATTERN: {exp.pattern}")
            summary_parts.append(f"SOLUTION: {exp.solution}")
            summary_parts.append(f"Applied successfully: {exp.application_count} times\n")

        return "\n".join(summary_parts)

    def record_experience_application(self, experience_id: str, success: bool) -> None:
        """Record when an experience is applied"""
        exp = next((e for e in self.experiences if e.experience_id == experience_id), None)
        if exp and success:
            exp.application_count += 1
            exp.confidence = min(1.0, exp.confidence + 0.05)  # Slight confidence boost
            self._save_experiences()
            logger.info(f"Recorded successful application of experience: {experience_id}")

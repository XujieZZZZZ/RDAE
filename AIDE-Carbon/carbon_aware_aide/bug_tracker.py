"""
Bug Tracker for AIDE-Carbon
Tracks and manages bug history to help LLM avoid repeating errors.
"""
import json
import logging
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from aide.journal import Node
from llm import call_llm

logger = logging.getLogger("aide")


@dataclass
class FixAttempt:
    """Single fix attempt record"""
    attempt_id: str
    fix_method: str  # Description of the fix method
    node_id: str  # Node where this fix was attempted
    edge_id: Optional[str]  # Edge ID for tracking
    timestamp: float
    success: bool  # Whether this fix worked

    def to_dict(self) -> dict:
        return {
            "attempt_id": self.attempt_id,
            "fix_method": self.fix_method,
            "node_id": self.node_id,
            "edge_id": self.edge_id,
            "timestamp": self.timestamp,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FixAttempt':
        return cls(**data)


@dataclass
class RawBugRecord:
    """Raw bug record - stores every error without merging"""
    raw_bug_id: str
    error_type: str
    error_message: str
    terminal_output: str
    code_snippet: str
    node_id: str
    parent_node_id: Optional[str]
    timestamp: float
    exc_info: Optional[dict]

    def to_dict(self) -> dict:
        return {
            "raw_bug_id": self.raw_bug_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "terminal_output": self.terminal_output,
            "code_snippet": self.code_snippet,
            "node_id": self.node_id,
            "parent_node_id": self.parent_node_id,
            "timestamp": self.timestamp,
            "exc_info": self.exc_info,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RawBugRecord':
        return cls(**data)


@dataclass
class BugRecord:
    """Single bug record with fix history"""
    bug_id: str
    error_type: str
    error_reason: str
    fix_suggestion: str
    error_code_snippet: str
    node_id: str
    parent_node_id: Optional[str]
    timestamp: float
    occurrence_count: int = 1
    summary: str = ""  # Short summary for LLM context
    successful_fixes: List[FixAttempt] = field(default_factory=list)  # Fixes that worked
    failed_fixes: List[FixAttempt] = field(default_factory=list)  # Fixes that didn't work

    def to_dict(self) -> dict:
        data = asdict(self)
        # Convert FixAttempt objects to dicts
        data['successful_fixes'] = [fix.to_dict() for fix in self.successful_fixes]
        data['failed_fixes'] = [fix.to_dict() for fix in self.failed_fixes]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'BugRecord':
        # Handle old records without new fields
        if 'summary' not in data:
            data['summary'] = ""
        if 'successful_fixes' not in data:
            data['successful_fixes'] = []
        if 'failed_fixes' not in data:
            data['failed_fixes'] = []

        # Convert dicts back to FixAttempt objects
        data['successful_fixes'] = [FixAttempt.from_dict(f) for f in data['successful_fixes']]
        data['failed_fixes'] = [FixAttempt.from_dict(f) for f in data['failed_fixes']]

        return cls(**data)


class BugTracker:
    """Bug tracking system"""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        self.bug_history_file = self.workspace_dir / "bug_history.json"
        self.raw_bug_history_file = self.workspace_dir / "raw_bug_history.json"
        self.bugs: List[BugRecord] = []
        self.raw_bugs: List[RawBugRecord] = []
        self.node_to_bug_map: Dict[str, str] = {}  # node_id -> bug_id mapping
        self._load_history()
        self._load_raw_history()

    def record_bug(self, node: Node) -> None:
        """Record a bug node"""
        try:
            # Step 1: Save raw bug record (no merging)
            raw_bug = self._save_raw_bug(node)

            # Step 2: Call LLM to analyze and update memory based on raw history
            result = self._analyze_and_update_memory(node, raw_bug)

            if result["should_update"] and result["target_bug_id"]:
                # Update existing bug memory
                similar_bug = next((b for b in self.bugs if b.bug_id == result["target_bug_id"]), None)
                if similar_bug:
                    # Update the bug record with new information
                    similar_bug.error_reason = result.get("updated_error_reason", similar_bug.error_reason)
                    similar_bug.fix_suggestion = result.get("updated_fix_suggestion", similar_bug.fix_suggestion)
                    similar_bug.summary = result.get("updated_summary", similar_bug.summary)
                    similar_bug.occurrence_count += 1
                    similar_bug.timestamp = time.time()

                    # Update node_to_bug_map for this node
                    self.node_to_bug_map[node.id] = similar_bug.bug_id

                    logger.info(f"Updated bug memory: {similar_bug.bug_id} (count: {similar_bug.occurrence_count}). Reason: {result.get('update_reasoning', 'N/A')}")
                else:
                    logger.warning(f"Target bug {result['target_bug_id']} not found, creating new bug")
                    result["should_update"] = False

            if not result["should_update"]:
                # Create new bug record
                bug_record = BugRecord(
                    bug_id=f"bug_{uuid.uuid4().hex[:8]}",
                    error_type=result.get("error_type", "UnknownError"),
                    error_reason=result.get("error_reason", "Unknown reason"),
                    fix_suggestion=result.get("fix_suggestion", "No suggestion available"),
                    error_code_snippet=result.get("error_code_snippet", ""),
                    node_id=node.id,
                    parent_node_id=node.parent.id if node.parent else None,
                    timestamp=time.time(),
                    summary=result.get("summary", ""),
                )
                self.bugs.append(bug_record)

                # Update node_to_bug_map for this node
                self.node_to_bug_map[node.id] = bug_record.bug_id

                logger.info(f"Recorded new bug: {bug_record.bug_id} ({bug_record.error_type})")

            # Save to file
            self._save_history()

        except Exception as e:
            logger.error(f"Failed to record bug: {e}")

    def get_bug_summary(self, max_bugs: int = 10) -> str:
        """Get concise bug summary for LLM with fix history"""
        if not self.bugs:
            return ""

        # Sort by occurrence count and timestamp
        sorted_bugs = sorted(
            self.bugs,
            key=lambda b: (b.occurrence_count, b.timestamp),
            reverse=True
        )[:max_bugs]

        summary_parts = ["【KNOWN BUGS - ROOT CAUSE ANALYSIS】"]

        for i, bug in enumerate(sorted_bugs, 1):
            summary_parts.append(f"\nBug #{i} ({bug.occurrence_count}x): {bug.summary}")
            summary_parts.append(f"ROOT CAUSE: {bug.error_reason}")
            summary_parts.append(f"FIX GUIDANCE: {bug.fix_suggestion}")

            # Show fix history based on whether bug was successfully resolved
            if bug.successful_fixes:
                # Bug was resolved - show ONLY the successful approach
                summary_parts.append(f"✓ RESOLVED: {bug.successful_fixes[-1].fix_method}")
            elif bug.failed_fixes:
                # Bug not resolved - show recent failed attempts as FORBIDDEN
                summary_parts.append(f"✗ UNRESOLVED - Failed attempts ({len(bug.failed_fixes)}):")
                for fix in bug.failed_fixes[-3:]:  # Show last 3 failed attempts
                    summary_parts.append(f"  - {fix.fix_method}")

        summary_parts.append("\n【DEBUGGING PROTOCOL】")
        summary_parts.append("1. Identify ROOT CAUSE (not symptom)")
        summary_parts.append("2. Check FORBIDDEN list above")
        summary_parts.append("3. Add diagnostic code if needed")
        summary_parts.append("4. Fix based on understanding")

        return "\n".join(summary_parts)

    def record_fix_attempt(self, node: Node, parent_bug_node: Node, success: bool) -> None:
        """
        Record a fix attempt for a bug

        Args:
            node: The node with the fix attempt
            parent_bug_node: The buggy parent node
            success: Whether the fix was successful
        """
        try:
            # Find the bug record for the parent node using the mapping
            bug_record = None
            bug_id = self.node_to_bug_map.get(parent_bug_node.id)

            if bug_id:
                # Found in mapping, get the bug record
                bug_record = next((b for b in self.bugs if b.bug_id == bug_id), None)
                if bug_record:
                    logger.info(f"Found bug record {bug_id} via node_to_bug_map for parent node {parent_bug_node.id}")

            # Fallback: try direct node_id match
            if not bug_record:
                bug_record = self._find_bug_by_node_id(parent_bug_node.id)
                if bug_record:
                    logger.info(f"Found bug record {bug_record.bug_id} via direct node_id match for {parent_bug_node.id}")
                    # Update the mapping for future use
                    self.node_to_bug_map[parent_bug_node.id] = bug_record.bug_id

            if not bug_record:
                logger.warning(f"No bug record found for node {parent_bug_node.id}, cannot record fix attempt")
                return

            # Get fix method from edge or node
            fix_method = "Unknown fix method"
            edge_id = None

            if hasattr(node, 'edge_to_parent') and node.edge_to_parent:
                edge = node.edge_to_parent
                edge_id = edge.edge_id

                # Try to get detailed summary from edge
                if hasattr(edge, 'detailed_summary') and edge.detailed_summary:
                    fix_method = edge.detailed_summary
                elif hasattr(edge, 'optimization_point') and edge.optimization_point:
                    fix_method = edge.optimization_point

            # Create fix attempt record
            fix_attempt = FixAttempt(
                attempt_id=f"fix_{uuid.uuid4().hex[:8]}",
                fix_method=fix_method,
                node_id=node.id,
                edge_id=edge_id,
                timestamp=time.time(),
                success=success,
            )

            # Add to appropriate list
            if success:
                bug_record.successful_fixes.append(fix_attempt)
                logger.info(f"Recorded successful fix for bug {bug_record.bug_id} (node: {node.id}, method: {fix_method[:100]})")
            else:
                bug_record.failed_fixes.append(fix_attempt)
                logger.info(f"Recorded failed fix attempt for bug {bug_record.bug_id} (node: {node.id}, method: {fix_method[:100]})")

            # Save to file immediately after recording
            self._save_history()

        except Exception as e:
            logger.error(f"Failed to record fix attempt: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _find_bug_by_node_id(self, node_id: str) -> Optional[BugRecord]:
        """Find bug record by node ID"""
        for bug in self.bugs:
            if bug.node_id == node_id:
                return bug
        return None

    def get_detailed_bug_info(self, max_bugs: int = 10) -> str:
        """Get detailed bug information (for debugging purposes)"""
        if not self.bugs:
            return ""

        # Sort by occurrence count and timestamp
        sorted_bugs = sorted(
            self.bugs,
            key=lambda b: (b.occurrence_count, b.timestamp),
            reverse=True
        )[:max_bugs]

        summary_parts = []
        for i, bug in enumerate(sorted_bugs, 1):
            summary_parts.append(f"""
Bug #{i} (ID: {bug.bug_id}, occurred {bug.occurrence_count} times):
- Type: {bug.error_type}
- Reason: {bug.error_reason}
- Fix: {bug.fix_suggestion}
- Code: {bug.error_code_snippet}
- Summary: {bug.summary}
""")

        return "\n".join(summary_parts)

    def get_similar_bugs(self, error_type: str, limit: int = 3) -> List[BugRecord]:
        """Get similar bugs"""
        similar = [b for b in self.bugs if b.error_type == error_type]
        return sorted(similar, key=lambda b: b.occurrence_count, reverse=True)[:limit]

    def _save_raw_bug(self, node: Node) -> RawBugRecord:
        """Save raw bug record without any merging"""
        # Get terminal output (last 2000 chars)
        term_out = node.term_out[-2000:] if node.term_out else "No output"

        # Get code snippet (last 500 chars)
        code_snippet = node.code[-500:] if len(node.code) > 500 else node.code

        raw_bug = RawBugRecord(
            raw_bug_id=f"raw_bug_{uuid.uuid4().hex[:8]}",
            error_type=node.exc_type or "UnknownError",
            error_message=str(node.exc_info) if node.exc_info else "No error message",
            terminal_output=term_out,
            code_snippet=code_snippet,
            node_id=node.id,
            parent_node_id=node.parent.id if node.parent else None,
            timestamp=time.time(),
            exc_info=node.exc_info,
        )

        self.raw_bugs.append(raw_bug)
        self._save_raw_history()
        logger.info(f"Saved raw bug: {raw_bug.raw_bug_id}")
        return raw_bug

    def _analyze_and_update_memory(self, node: Node, raw_bug: RawBugRecord) -> Dict[str, any]:
        """
        Analyze bug and update memory based on raw history
        Returns information needed to update or create bug memory
        """

        # Get recent raw bugs related to this error type (last 5)
        related_raw_bugs = [
            rb for rb in self.raw_bugs[-10:]
            if rb.error_type == raw_bug.error_type or
               any(keyword in rb.error_message for keyword in raw_bug.error_message.split()[:3])
        ][-5:]

        # Prepare raw bug history
        raw_history_info = []
        for rb in related_raw_bugs:
            raw_history_info.append({
                "raw_bug_id": rb.raw_bug_id,
                "error_type": rb.error_type,
                "error_message": rb.error_message[:200],
                "terminal_output": rb.terminal_output[:300],
                "timestamp": rb.timestamp,
            })

        raw_history_json = json.dumps(raw_history_info, indent=2) if raw_history_info else "[]"

        # Prepare existing bug memories
        existing_bugs_info = []
        for bug in self.bugs:
            existing_bugs_info.append({
                "bug_id": bug.bug_id,
                "error_type": bug.error_type,
                "error_reason": bug.error_reason,
                "fix_suggestion": bug.fix_suggestion,
                "occurrence_count": bug.occurrence_count,
                "summary": bug.summary,
                "failed_fixes": [f.fix_method for f in bug.failed_fixes[-3:]]
            })

        existing_bugs_json = json.dumps(existing_bugs_info, indent=2) if existing_bugs_info else "[]"

        prompt = f"""
You are an expert in debugging and learning from repeated errors.

【TASK】
Analyze the new bug and decide whether to UPDATE an existing bug memory or CREATE a new one.

CRITICAL PRINCIPLE: Focus on ROOT CAUSE analysis, not symptom treatment.

IMPORTANT: When the same type of error occurs repeatedly with different attempted fixes, you should UPDATE the existing memory to reflect the accumulated knowledge, NOT create a new bug.

【NEW BUG】
Error Type: {raw_bug.error_type}
Error Message: {raw_bug.error_message}
Terminal Output (last 300 chars):
{raw_bug.terminal_output[-300:]}
Code (last 200 chars):
{raw_bug.code_snippet[-200:]}

【RECENT RAW BUG HISTORY (same error type)】
{raw_history_json}

【EXISTING BUG MEMORIES】
{existing_bugs_json}

【ROOT CAUSE ANALYSIS GUIDELINES】

When analyzing errors, identify the FUNDAMENTAL PROBLEM, not just the symptom:

1. **API Misunderstanding Errors** (e.g., AttributeError, KeyError on framework objects)
   - ROOT CAUSE: The code assumes an API that doesn't exist
   - CORRECT ANALYSIS: Identify that the actual API is unknown; list all attempted attribute/method names that have failed
   - FIX SUGGESTION FORMAT: "FORBIDDEN: [list all failed attempts]. REQUIRED: Add debug code to inspect the actual API using dir(object) or check framework documentation before any fix attempt."

2. **Logic Errors** (e.g., AssertionError, ValueError with computed values)
   - ROOT CAUSE: The computation logic produces invalid values
   - CORRECT ANALYSIS: Identify which computation step produces the invalid value and why
   - FIX SUGGESTION FORMAT: "FORBIDDEN: Do not force values to be valid without understanding why they're invalid. REQUIRED: Add logging to trace the computation: print intermediate values to identify where the value becomes invalid."

3. **Type Confusion Errors** (e.g., treating list as dict, or vice versa)
   - ROOT CAUSE: Incorrect assumption about data structure type
   - CORRECT ANALYSIS: Identify the assumed type vs actual type
   - FIX SUGGESTION FORMAT: "REQUIRED: Add type checking to determine actual type before accessing."

【MEMORY UPDATE CRITERIA】
You should UPDATE an existing bug memory if:
1. The new error has the SAME ROOT CAUSE as an existing bug
2. The error represents a FAILED FIX ATTEMPT for a known issue
3. Multiple attempts show a PATTERN (e.g., multiple different attribute names all failing)

When updating, you should:
- GENERALIZE the error_reason to cover all observed cases and identify the ROOT CAUSE
- UPDATE fix_suggestion to:
  * List ALL failed attempts as FORBIDDEN approaches
  * Provide REQUIRED diagnostic steps (not guesses)
  * Emphasize understanding before fixing
- ENHANCE summary to include accumulated knowledge

【OUTPUT FORMAT】
Respond with ONLY valid JSON (no markdown, no code blocks):

{{
  "error_type": "short category name",
  "error_reason": "ROOT CAUSE explanation (not just symptom), generalized if updating",
  "fix_suggestion": "FORBIDDEN: [list failed attempts]. REQUIRED: [diagnostic steps to understand root cause]",
  "error_code_snippet": "specific code causing error (max 5 lines)",
  "should_update": true or false,
  "target_bug_id": "bug_xxxxx" or null,
  "update_reasoning": "why updating/creating (1 sentence)",
  "summary": "one-sentence summary emphasizing ROOT CAUSE and required diagnostic approach",
  "updated_error_reason": "if updating, the new generalized ROOT CAUSE",
  "updated_fix_suggestion": "if updating, the new suggestion with FORBIDDEN list and REQUIRED diagnostics",
  "updated_summary": "if updating, the new summary emphasizing ROOT CAUSE"
}}

CRITICAL RULES:
1. Output ONLY JSON, no markdown
2. If should_update is true, you MUST provide updated_error_reason, updated_fix_suggestion, and updated_summary
3. If should_update is false, set target_bug_id to null
4. When updating, incorporate knowledge from failed_fixes in the existing bug
5. Generalize patterns and identify ROOT CAUSE (e.g., "API unknown" not "no specific attribute")
6. Fix suggestions must include FORBIDDEN (what not to try) and REQUIRED (diagnostic steps)
7. Emphasize UNDERSTANDING the problem before attempting fixes
"""

        try:
            response = call_llm(prompt)

            # Clean up response
            response = response.strip()
            if response.startswith("```"):
                lines = response.split('\n')
                start_idx = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith("```"):
                        start_idx = i + 1
                    elif line.strip().startswith("{"):
                        start_idx = i
                        break

                end_idx = len(lines)
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().startswith("```"):
                        end_idx = i
                    elif lines[i].strip().endswith("}"):
                        end_idx = i + 1
                        break

                response = '\n'.join(lines[start_idx:end_idx])

            # Parse JSON
            result = json.loads(response)

            # Validate
            if result["should_update"] and not result["target_bug_id"]:
                logger.warning("LLM set should_update=true but target_bug_id is null, correcting")
                result["should_update"] = False

            if not result["should_update"] and result["target_bug_id"]:
                logger.warning("LLM set should_update=false but provided target_bug_id, correcting")
                result["target_bug_id"] = None

            logger.info(f"LLM analysis: {result['error_type']}, update={result['should_update']}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response: {response[:500]}")
            return self._fallback_analysis(node, raw_bug.code_snippet)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analysis(node, raw_bug.code_snippet)

    def _fallback_analysis(self, node: Node, code_snippet: str) -> Dict[str, any]:
        """Fallback analysis when LLM call fails"""
        return {
            "error_type": node.exc_type or "UnknownError",
            "error_reason": "Failed to analyze with LLM",
            "fix_suggestion": "Check the error manually",
            "error_code_snippet": code_snippet[:200],
            "should_update": False,
            "target_bug_id": None,
            "update_reasoning": "LLM analysis failed, creating new bug",
            "summary": f"Error: {node.exc_type or 'Unknown'} - manual review needed",
            "updated_error_reason": "",
            "updated_fix_suggestion": "",
            "updated_summary": ""
        }

    def _load_history(self) -> None:
        """Load bug history from file"""
        if not self.bug_history_file.exists():
            logger.info("No existing bug history file")
            return

        try:
            with open(self.bug_history_file, 'r') as f:
                data = json.load(f)
                self.bugs = [BugRecord.from_dict(b) for b in data.get("bugs", [])]
            logger.info(f"Loaded {len(self.bugs)} bugs from history")

            # Rebuild node_to_bug_map from loaded bugs
            self._rebuild_node_to_bug_map()

        except Exception as e:
            logger.error(f"Failed to load bug history: {e}")
            self.bugs = []

    def _load_raw_history(self) -> None:
        """Load raw bug history from file"""
        if not self.raw_bug_history_file.exists():
            logger.info("No existing raw bug history file")
            return

        try:
            with open(self.raw_bug_history_file, 'r') as f:
                data = json.load(f)
                self.raw_bugs = [RawBugRecord.from_dict(b) for b in data.get("raw_bugs", [])]
            logger.info(f"Loaded {len(self.raw_bugs)} raw bugs from history")
        except Exception as e:
            logger.error(f"Failed to load raw bug history: {e}")
            self.raw_bugs = []

    def _save_history(self) -> None:
        """Save bug history to file"""
        try:
            # Ensure directory exists
            self.workspace_dir.mkdir(parents=True, exist_ok=True)

            data = {
                "bugs": [bug.to_dict() for bug in self.bugs],
                "metadata": {
                    "total_bugs": len(self.bugs),
                    "last_updated": time.time()
                }
            }

            with open(self.bug_history_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.bugs)} bugs to {self.bug_history_file}")
        except Exception as e:
            logger.error(f"Failed to save bug history: {e}")

    def _save_raw_history(self) -> None:
        """Save raw bug history to file"""
        try:
            # Ensure directory exists
            self.workspace_dir.mkdir(parents=True, exist_ok=True)

            data = {
                "raw_bugs": [bug.to_dict() for bug in self.raw_bugs],
                "metadata": {
                    "total_raw_bugs": len(self.raw_bugs),
                    "last_updated": time.time()
                }
            }

            with open(self.raw_bug_history_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.raw_bugs)} raw bugs to {self.raw_bug_history_file}")
        except Exception as e:
            logger.error(f"Failed to save raw bug history: {e}")

    def _rebuild_node_to_bug_map(self) -> None:
        """Rebuild node_to_bug_map from loaded bug records"""
        self.node_to_bug_map.clear()

        for bug in self.bugs:
            # Map the original bug node
            self.node_to_bug_map[bug.node_id] = bug.bug_id

            # Map all nodes from failed fixes
            for fix in bug.failed_fixes:
                self.node_to_bug_map[fix.node_id] = bug.bug_id

            # Map all nodes from successful fixes
            for fix in bug.successful_fixes:
                self.node_to_bug_map[fix.node_id] = bug.bug_id

        logger.info(f"Rebuilt node_to_bug_map with {len(self.node_to_bug_map)} entries")


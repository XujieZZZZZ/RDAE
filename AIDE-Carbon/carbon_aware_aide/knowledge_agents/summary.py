from llm import call_llm
from typing import Dict, List, Optional


def summarization_module(prompt, model=None, temperature=0.7):
    """
    General summarization module using LLM.

    Args:
        prompt: The prompt or content to summarize
        model: Model name (optional)
        temperature: Temperature for generation

    Returns:
        Summarized text
    """
    completion_text = call_llm(
        prompt=prompt,
        # user_message=None,
        # model=model,
        # temperature=temperature,
    )
    return completion_text


def extract_task_decomposition(task_data: Dict) -> Optional[Dict]:
    """
    Extract task decomposition information from task data using LLM.

    Args:
        task_data: Full task data containing task information

    Returns:
        Dictionary containing decomposition steps, or None if extraction fails
    """
    if not task_data:
        return None

    # First, try to extract any existing structured fields as fallback
    fallback_decomposition = {}
    if "decomposition" in task_data:
        fallback_decomposition["steps"] = task_data["decomposition"]
    if "algorithm_steps" in task_data:
        fallback_decomposition["algorithm_steps"] = task_data["algorithm_steps"]
    if "implementation_plan" in task_data:
        fallback_decomposition["implementation_plan"] = task_data["implementation_plan"]
    if "constraints" in task_data:
        fallback_decomposition["constraints"] = task_data["constraints"]
    if "summary_dict" in task_data and task_data["summary_dict"]:
        fallback_decomposition["task_summary"] = task_data["summary_dict"].get("summary", "")

    # Try LLM-based extraction for richer information
    try:
        # Convert task_data to a readable format (limit size to avoid token overflow)
        import json
        task_data_str = json.dumps(task_data, indent=2, ensure_ascii=False)

        # If task_data is too large, truncate it
        max_length = 8000  # characters
        if len(task_data_str) > max_length:
            task_data_str = task_data_str[:max_length] + "\n... [truncated]"

        prompt = f"""You are an expert in analyzing task decomposition from task execution data.

Given the following task data, extract and summarize the task decomposition approach used.

TASK DATA:
{task_data_str}

Analyze this data and extract:
1. **Task Description**: What problem was being solved?
2. **Decomposition Steps**: How was the task broken down into steps?
3. **Algorithm Approach**: What algorithmic strategy was used?
4. **Implementation Plan**: What was the implementation approach?
5. **Constraints**: What constraints or requirements were considered?
6. **Key Decisions**: What were the critical decision points?

If the data contains code, analyze the code structure and logic to infer the decomposition.
If the data contains execution results, use them to understand the approach.

Output a structured summary in the following format:

TASK DESCRIPTION:
[Brief description of the task]

DECOMPOSITION STEPS:
1. [Step 1]
2. [Step 2]
...

ALGORITHM APPROACH:
[Description of the algorithmic strategy]

IMPLEMENTATION PLAN:
[Key implementation details]

CONSTRAINTS:
- [Constraint 1]
- [Constraint 2]
...

KEY DECISIONS:
- [Decision 1]
- [Decision 2]
...

Be concise but comprehensive. Focus on the decomposition strategy that can be learned from this example.
"""

        decomposition_text = summarization_module(prompt, temperature=0.3)

        # Parse the structured output into a dictionary
        decomposition = {
            "raw_text": decomposition_text,
            "task_data_summary": task_data.get("summary_dict", {}).get("summary", "")
        }

        # Try to extract structured sections from the response
        sections = {
            "TASK DESCRIPTION:": "task_description",
            "DECOMPOSITION STEPS:": "steps",
            "ALGORITHM APPROACH:": "algorithm_approach",
            "IMPLEMENTATION PLAN:": "implementation_plan",
            "CONSTRAINTS:": "constraints",
            "KEY DECISIONS:": "key_decisions"
        }

        for section_header, section_key in sections.items():
            if section_header in decomposition_text:
                start_idx = decomposition_text.find(section_header) + len(section_header)
                # Find the next section or end of text
                end_idx = len(decomposition_text)
                for other_header in sections.keys():
                    if other_header != section_header:
                        next_idx = decomposition_text.find(other_header, start_idx)
                        if next_idx != -1 and next_idx < end_idx:
                            end_idx = next_idx

                section_content = decomposition_text[start_idx:end_idx].strip()
                if section_content:  # Only add non-empty sections
                    decomposition[section_key] = section_content

        # If LLM extraction succeeded and has content, return it
        if len(decomposition) > 2:  # More than just raw_text and task_data_summary
            return decomposition

        # Otherwise, fall back to structured fields
        return fallback_decomposition if fallback_decomposition else None

    except Exception:
        # If LLM extraction fails, return fallback
        return fallback_decomposition if fallback_decomposition else None


def summarize_decomposition_patterns(patterns: List[Dict], user_task: str) -> str:
    """
    Summarize multiple task decomposition patterns into a coherent guide.

    Args:
        patterns: List of decomposition patterns from similar tasks
        user_task: User's task description

    Returns:
        Summarized decomposition guide
    """
    if not patterns:
        # If no patterns available, provide a generic decomposition guide
        return f"""No similar task decomposition patterns were found in the knowledge base.

For the given task:
{user_task}

Consider a general decomposition approach:
1. **Problem Analysis**: Break down the core objective into sub-goals
2. **Constraint Identification**: List all constraints and invariants that must be maintained
3. **Algorithm Design**: Design the core algorithmic approach and decision logic
4. **Component Decomposition**: Identify key components and their interfaces
5. **Implementation Planning**: Order implementation steps based on dependencies
6. **Testing Strategy**: Plan how to validate each component and the integrated system

This is a generic framework. Adapt it based on the specific requirements of your task."""

    # Build prompt for summarization
    prompt = f"""You are an expert in task decomposition and algorithm design.

Given the following similar task decomposition patterns, synthesize a comprehensive decomposition strategy for the user's task.

USER'S TASK:
{user_task}

SIMILAR TASK DECOMPOSITION PATTERNS:
"""

    # Add patterns with available information
    valid_patterns = 0
    for i, pattern in enumerate(patterns, 1):
        decomposition = pattern.get('decomposition', {})

        # Check if decomposition has any useful content
        has_content = any(
            key in decomposition and decomposition[key]
            for key in ['steps', 'algorithm_approach', 'implementation_plan', 'constraints',
                       'task_description', 'key_decisions', 'raw_text']
        )

        if not has_content:
            continue

        valid_patterns += 1
        prompt += f"\n--- Pattern {valid_patterns} (Similarity: {pattern.get('similarity_score', 0):.3f}) ---\n"
        prompt += f"Task Summary: {pattern.get('task_summary', 'N/A')}\n"

        # Add available decomposition information
        if 'task_description' in decomposition and decomposition['task_description']:
            prompt += f"Task Description: {decomposition['task_description']}\n"

        if 'steps' in decomposition and decomposition['steps']:
            prompt += f"Decomposition Steps:\n{decomposition['steps']}\n"

        if 'algorithm_approach' in decomposition and decomposition['algorithm_approach']:
            prompt += f"Algorithm Approach:\n{decomposition['algorithm_approach']}\n"

        if 'implementation_plan' in decomposition and decomposition['implementation_plan']:
            prompt += f"Implementation Plan:\n{decomposition['implementation_plan']}\n"

        if 'constraints' in decomposition and decomposition['constraints']:
            prompt += f"Constraints:\n{decomposition['constraints']}\n"

        if 'key_decisions' in decomposition and decomposition['key_decisions']:
            prompt += f"Key Decisions:\n{decomposition['key_decisions']}\n"

        # If no structured fields, use raw_text
        if valid_patterns == i and 'raw_text' in decomposition and decomposition['raw_text']:
            prompt += f"Analysis:\n{decomposition['raw_text'][:1000]}\n"  # Limit length

    # If no valid patterns after filtering, return generic guide
    if valid_patterns == 0:
        return f"""The retrieved similar tasks did not contain sufficient decomposition information.

For the given task:
{user_task}

Recommend following a structured decomposition approach:
1. Analyze the problem domain and identify key challenges
2. Define clear objectives and success criteria
3. Identify constraints and invariants
4. Design the core algorithm or approach
5. Break down into implementable components
6. Plan implementation order based on dependencies
7. Define testing and validation strategy"""

    prompt += f"""

Based on these {valid_patterns} pattern(s), provide a synthesized task decomposition strategy that:
1. Identifies common decomposition approaches across similar tasks
2. Highlights key steps and phases
3. Notes important constraints and considerations
4. Adapts the patterns to fit the user's specific task

Output a clear, structured decomposition guide."""

    try:
        summary = summarization_module(prompt, temperature=0.3)
        return summary
    except Exception:
        # Fallback: return a summary of what we have
        return f"""Based on {valid_patterns} similar task(s), here are key insights:

{chr(10).join([f"- Pattern {i+1}: {p.get('task_summary', 'N/A')[:200]}..." for i, p in enumerate(patterns[:3])])}

Apply similar decomposition strategies to your task, focusing on:
1. Breaking down the problem into manageable sub-tasks
2. Identifying critical constraints and dependencies
3. Designing a clear algorithmic approach
4. Planning implementation in logical phases"""


def formalize_user_requirement(user_requirement: str, task_context: str = "", reference_formalizations: List[Dict] = None) -> str:
    """
    Formalize user requirement into a structured task description.

    Args:
        user_requirement: Raw user requirement text
        task_context: Additional context about the task domain
        reference_formalizations: Optional list of formalization examples from similar tasks

    Returns:
        Formalized task description
    """
    task_context = task_context.strip() if task_context else ""
    prompt = f"""You are an expert in formalizing software engineering requirements.

Given a user requirement, formalize it into a clear, structured task description that can be used for:
1. Retrieving similar tasks from a knowledge base
2. Guiding task decomposition
3. Informing code generation

USER REQUIREMENT:
{user_requirement}

"TASK CONTEXT:\n{task_context}\n"""

    # Add reference formalizations if provided
    if reference_formalizations:
        prompt += "\n\nREFERENCE FORMALIZATION EXAMPLES (from similar tasks):\n"
        for i, ref in enumerate(reference_formalizations, 1):
            prompt += f"\n--- Example {i} (Similarity: {ref.get('similarity_score', 0):.3f}) ---\n"
            prompt += f"Original Task: {ref.get('original_task', 'N/A')}\n"
            prompt += f"Formalized Version:\n{ref.get('formalization', 'N/A')}\n"

        prompt += "\n\nLearn from these examples how to structure and formalize requirements in this domain."

    prompt += """

Formalize this requirement by:
1. Identifying the core objective
2. Extracting key constraints and requirements
3. Clarifying the input/output specifications
4. Noting any performance or quality metrics
5. Highlighting domain-specific considerations

Output a structured, formal task description."""

    formalized = summarization_module(prompt, temperature=0.2)
    return formalized


def extract_formalization_from_task(task_data: Dict) -> Optional[Dict]:
    """
    Extract formalization approach from task data using LLM.

    Args:
        task_data: Full task data

    Returns:
        Dictionary containing original task and its formalization
    """
    if not task_data:
        return None

    import json
    task_data_str = json.dumps(task_data, indent=2, ensure_ascii=False)

    prompt = f"""You are an expert in analyzing how tasks are formalized.

Given the following task data, extract:
1. The original/informal task description (if available)
2. The formalized/structured version of the task

TASK DATA:
{task_data_str}

Analyze this data and identify:
- What was the original problem statement or requirement?
- How was it formalized into a structured description?
- What key elements were emphasized in the formalization?

Output in the following format:

ORIGINAL TASK:
[The original/informal task description]

FORMALIZED TASK:
[The structured/formal version]

KEY FORMALIZATION ASPECTS:
- [Aspect 1: e.g., "Emphasized performance metrics"]
- [Aspect 2: e.g., "Clearly defined constraints"]
...

If you cannot find distinct original and formalized versions, extract the most structured description available and note the key aspects of how it's structured.
"""

    try:
        result_text = summarization_module(prompt, temperature=0.3)

        formalization = {
            "raw_text": result_text
        }

        # Extract sections
        if "ORIGINAL TASK:" in result_text:
            start_idx = result_text.find("ORIGINAL TASK:") + len("ORIGINAL TASK:")
            end_idx = result_text.find("FORMALIZED TASK:") if "FORMALIZED TASK:" in result_text else len(result_text)
            formalization["original_task"] = result_text[start_idx:end_idx].strip()

        if "FORMALIZED TASK:" in result_text:
            start_idx = result_text.find("FORMALIZED TASK:") + len("FORMALIZED TASK:")
            end_idx = result_text.find("KEY FORMALIZATION ASPECTS:") if "KEY FORMALIZATION ASPECTS:" in result_text else len(result_text)
            formalization["formalization"] = result_text[start_idx:end_idx].strip()

        if "KEY FORMALIZATION ASPECTS:" in result_text:
            start_idx = result_text.find("KEY FORMALIZATION ASPECTS:") + len("KEY FORMALIZATION ASPECTS:")
            formalization["key_aspects"] = result_text[start_idx:].strip()

        # Fallback: use summary if available
        if not formalization.get("formalization") and "summary_dict" in task_data:
            formalization["formalization"] = task_data["summary_dict"].get("summary", "")

        return formalization

    except Exception as e:
        # Fallback: try to use summary
        if "summary_dict" in task_data and task_data["summary_dict"]:
            return {
                "formalization": task_data["summary_dict"].get("summary", ""),
                "original_task": task_data.get("task_description", "")
            }
        return None
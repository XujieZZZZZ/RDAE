import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
import sys
import os
import textwrap

# Add parent directory to path to import llm module
from llm import call_llm

# from retrieval_agent import RetrievalQwenWithSummarizationAgent


class TaskDecompositionAgent:
    """
    Agent for decomposing user tasks using retrieval-augmented generation.
    """
    def __init__(
        self,
        retrieval_agent,
        task_context_file: Optional[str] = None,
        top_k: int = 3,
    ):
        """
        Initialize the task decomposition agent.

        Args:
            knowledge_base_files: List of file paths containing similar task examples
            task_context_file: Optional file containing task domain context
            top_k: Number of similar tasks to retrieve
        """
        self.retrieval_agent = retrieval_agent
        # self.retrieval_agent = RetrievalQwenWithSummarizationAgent(
        #     file_list=knowledge_base_files,
        #     top_k=top_k
        # )
        # self.task_context = self._load_task_context(task_context_file)
        # self.top_k = top_k

    def _load_task_context(self, context_file: Optional[str]) -> str:
        """Load task context from file if provided."""
        if context_file and Path(context_file).exists():
            with open(context_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def decompose_task(self, task: str, simulator_info: Optional[str] = None) -> Dict:
        """
        Decompose a research task into high-level subtasks.

        Args:
            task: User's research task description
            simulator_info: Optional information about the simulator environment and constraints

        Returns:
            Dict containing:
                - user_requirement: original task
                - formalization_references: similar tasks retrieved
                - formalized_task: formalized description
                - similar_tasks: similar decomposition examples
                - decomposition_guide: synthesis of decomposition strategies
                - final_decomposition: the actual decomposition with steps and components
        """
        # Step 1: Retrieve similar tasks and decomposition examples
        similar_examples = self.retrieval_agent.retrieve(task)

        # Step 2: Extract decomposition patterns from similar examples
        formalization_references = []
        similar_tasks = []

        for example in similar_examples:
            if example.get("summary") and example.get("formalization"):
                formalization_references.append({
                    "summary": example["summary"],
                    "similarity_score": example.get("similarity_score", 0.0),
                    "formalization": example["formalization"],
                    "original_task": example.get("original_task", "")
                })

            if example.get("summary"):
                similar_tasks.append({
                    "summary": example["summary"],
                    "similarity_score": example.get("similarity_score", 0.0)
                })

        # Step 3: Build prompt for task formalization (with simulator info)
        formalization_prompt = self._build_formalization_prompt(task, formalization_references, simulator_info)

        # Step 4: Call LLM to formalize the task
        formalized_task = call_llm(formalization_prompt)

        # Step 5: Build prompt for final decomposition (directly from formalization and similar tasks)
        decomposition_prompt = self._build_decomposition_prompt(
            task, formalized_task, similar_tasks
        )

        # Step 6: Call LLM to generate final decomposition
        final_decomposition_str = call_llm(decomposition_prompt)

        return final_decomposition_str

        # Step 9: Parse the final decomposition (assuming JSON format)
        try:
            import json
            final_decomposition = json.loads(final_decomposition_str)
        except:
            final_decomposition = {
                "strategy": final_decomposition_str,
                "steps": [],
                "components": []
            }

        # Step 7: Construct result
        result = {
            "user_requirement": task,
            "formalization_references": formalization_references,
            "formalized_task": formalized_task,
            "similar_tasks": similar_tasks,
            "final_decomposition": final_decomposition
        }

        return result

    def pre_simulator_info_for_formulation(self, simulator_info: str) -> str:
        prompt = f"""
You are an expert research assistant. Your task is to summarize a simulator environment for the purpose of problem formalization.

Simulator Info (raw): {simulator_info}

Instructions:
1. Read the provided simulator information carefully.
2. Extract only the information relevant for understanding the research problem at a high level, ignoring implementation details such as:
   - Class names, method signatures
   - Field names in objects or dictionaries
   - Programming interface details
3. Identify and describe:
   - **Abstract Inputs**: What types of information the algorithm observes or receives from the simulator.
   - **Abstract Outputs**: What decisions or actions the algorithm is expected to produce.
   - **Constraints**: High-level limitations imposed by the simulator (e.g., resource limits, timing, correctness conditions).
   - **Performance Requirements**: Any simulator-enforced requirements that affect the solution's success.
4. Present the output as a concise structured summary in Markdown:
   - High-Level Simulator Overview
   - Inputs
   - Outputs
   - Constraints
   - Performance Requirements
5. Do not include any low-level implementation specifics.

The goal is to produce a simulator description suitable for **problem formalization**, not for coding or implementation.
"""
        simulator_for_formulation = call_llm(prompt)
        return simulator_for_formulation

    def _build_formalization_prompt(self, task: str, references: List[Dict], simulator_info: Optional[str] = None) -> str:
        """
        Build a structured prompt for task formalization.

        Args:
            task: The new research task that needs formalization.
            references: List of similar tasks with their formalizations.
                        Each item should be a dict with 'original_task' and 'formalization' keys.
            simulator_info: Optional information about the simulator environment, constraints, and interface.

        Returns:
            A string prompt for the LLM.
        """

        # Prepare the reference section
        reference_texts = []
        for i, ref in enumerate(references, 1):
            ref_text = [
                f"{i}. Original Task: {ref['original_task']}\n   Formalization: {ref['formalization']}"
                for i, ref in enumerate(references, 1)
            ]
            reference_texts.append(ref_text)

        references_section = "\n".join(reference_texts)
        if len(references_section) == 0:
            references_section = "No similar tasks found in the knowledge base."

        # Prepare the simulator information section
        simulator_section = ""
        if simulator_info:
            simulator_section = f"""Simulator Environment Information:
{simulator_info}

IMPORTANT: Your formalization must account for the simulator's constraints, interface requirements, and execution environment described above. Consider how the solution will interact with the simulator and what technical constraints must be satisfied.
"""

        """Build prompt for task formalization."""
        prompt = f"""
You are an expert research assistant. Your goal is to produce a **clear, concise, and structured formalization** of a research task, suitable for research-level documentation and guiding algorithm design.

User's Task: {task}

Similar Task Formalizations (references): {references_section}

Simulator Overview (abstracted, optional): {simulator_section}

Instructions:
1. Analyze the task deeply, focusing on:
   - What must be solved
   - Why it is challenging
   - What counts as success

2. Identify and describe:
   - **Inputs**: All information needed to make decisions, including any high-level simulator-provided observations.
   - **Outputs**: Decisions or actions the algorithm must produce.
   - **Constraints**: Limitations or conditions inherent to the problem or imposed by the simulator (abstract, without implementation details).
   - **Assumptions and edge cases**: Implicit assumptions or unusual conditions that affect problem formulation.
   - **Optimization Objective**: Include a concise **formula** that captures the trade-off between performance metrics and any other cost (e.g., carbon footprint), using abstract symbols if necessary. Do not include low-level implementation formulas or field names.

3. Provide a structured formalization with these sections:
   - High-Level Objective and Core Challenge
   - Inputs
   - Outputs
   - Optimization Objective (with formula)
   - Constraints
   - Success Criteria (including abstract simulator validation if applicable)

4. Avoid detailed simulator implementation, class names, method signatures, object fields, or programming-specific details.

5. Ensure the output is:
   - Concise and precise
   - Comprehensive, capturing all critical aspects of the research problem
   - Written in professional academic style

The output should focus on **problem abstraction, optimization goals, and success measures**, not coding or simulator-level implementation.
"""

        return prompt

    def _build_decomposition_prompt(self, task: str, formalized_task: str,
                                   similar_tasks: List[Dict]=None) -> str:
        """Build prompt for final task decomposition."""

        # Build similar tasks section
        similar_section = "\n".join(
            f"{i}. {sim['summary'][:800]}..." for i, sim in enumerate(similar_tasks, 1)
        )

        prompt = f"""You are an expert research assistant. Your task is to decompose a complex research problem into smaller, **solvable research subproblems** that can each produce concrete solutions, algorithms, or constructions. Avoid creating subproblems that are only evaluative, analytical, or verification-focused.

User's Task:
{task}

Formalized Task:
{formalized_task}

Similar Task Examples:
{similar_section}

Instructions for high-quality decomposition:
1. Analyze the formalized task to identify the main challenges, goals, and constraints.
2. Examine similar task examples and extract decomposition strategies.
3. Decompose the task into 2~4 high-level, solvable research subproblems:
   - Each subtask should be a **problem that can be solved** (algorithm, construction, or method) contributing toward the main task.
   - Avoid subtasks that only "measure", "verify", or "analyze"; instead, focus on subtasks that produce a tangible solution or output.
   - Simplify, isolate, or clarify specific challenges in the main task to make them tractable.
   - Maintain logical dependencies where applicable.
4. For each subtask, provide:
   - "description": the solvable subproblem statement.
   - "details": assumptions, inputs, expected outputs, and criteria for success.
   - "dependencies": list of subtask numbers this depends on, if any.
5. Optionally, define research components that support solving multiple subtasks, describing their purpose and focus.
6. Summarize the overall decomposition strategy in "strategy".
7. Ensure completeness: all critical challenges in the formalized task are addressed in at least one subtask.
8. Avoid implementation-level instructions; focus on solvable research subproblems.

Output format (JSON only):

{{
    "strategy": "Brief description of decomposition logic and reasoning",
    "subtasks": [
        {{
            "subtask_number": 1,
            "description": "Solvable subproblem statement",
            "details": "Assumptions, inputs, expected outputs, criteria for success",
            "dependencies": [list of subtask numbers]
        }}
    ]
}}
"""

        return prompt
    

# with open("/Users/dingch/Desktop/Projections/AI4Research/aideml-main-FIFO/aideml-main/subtask_agents/user_requirement.md", "r", encoding="utf-8") as rf:
#     task_description = rf.read()
# with open("/Users/dingch/Desktop/Projections/AI4Research/aideml-main-FIFO/aideml-main/subtask_agents/simulator_info.md", "r", encoding="utf-8") as rf:
#     simulator_info = rf.read()
# task_agent = TaskDecompositionAgent(knowledge_base_files=[""])
# simulator_for_formulation = task_agent.pre_simulator_info_for_formulation(simulator_info=simulator_info)
# formalization_prompt = task_agent._build_formalization_prompt(task=task_description, references=[], simulator_info=simulator_for_formulation)
# print(formalization_prompt, "\n\n", "="*80, "\n\n")
# formalized_task = call_llm(formalization_prompt)
# print(formalized_task)


# with open("/Users/dingch/Desktop/Projections/AI4Research/aideml-main-FIFO/aideml-main/subtask_agents/user_requirement.md", "r", encoding="utf-8") as rf:
#     task_description = rf.read()
# with open("/Users/dingch/Desktop/Projections/AI4Research/aideml-main-FIFO/aideml-main/subtask_agents/results/results_with_simulator/formulation_provide_by_LLM.md", "r", encoding="utf-8") as rf:
#     formalized_task = rf.read()
# task_agent = TaskDecompositionAgent(knowledge_base_files=[""])

# decomposition_prompt = task_agent._build_decomposition_prompt(task=task_description, formalized_task=formalized_task, similar_tasks=[])
# print(decomposition_prompt)
# final_decomposition_str = call_llm(decomposition_prompt)
# print(final_decomposition_str)
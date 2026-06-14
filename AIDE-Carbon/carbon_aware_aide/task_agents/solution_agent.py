from typing import Dict, List, Optional
from llm import call_llm
import json
import os
from task_agents.task_agent import TaskDecompositionAgent


class SolutionAgent:
    """Agent for generating implementation guidance."""

    def __init__(
        self, 
        simulator_info_file: Optional[str] = None,
        user_requirement_file: Optional[str] = None,
        retrieval_agent = None,
    ):
        """Initialize with optional retrieval agent."""
        with open(user_requirement_file, "r", encoding="utf-8") as rf:
            self.user_requirement = rf.read()
        with open(simulator_info_file, "r", encoding="utf-8") as rf:
            self.simulator_info = rf.read()

        self.task_agent = TaskDecompositionAgent(
            retrieval_agent = retrieval_agent,
        )
        self.retrieval_agent = retrieval_agent
    
    def generate_formalization_subtasks(self, task_description: str=None, simulator_info: str=None):
        simulator_for_formulation = self.task_agent.pre_simulator_info_for_formulation(
            simulator_info=simulator_info if simulator_info is not None else self.simulator_info
        )
        formalization_prompt = self.task_agent._build_formalization_prompt(
            task=task_description if task_description is not None else self.user_requirement,
            references=[],
            simulator_info=simulator_for_formulation
        )
        formalization = call_llm(formalization_prompt)
        decomposition_prompt = self.task_agent._build_decomposition_prompt(
            task=task_description if task_description is not None else self.user_requirement,
            formalized_task=formalization,
            similar_tasks=[]
        )
        subtasks = call_llm(decomposition_prompt)
        
        return subtasks, formalization, self.simulator_info

    def generate_all_solutions(
        self,
        subtasks: List[Dict],
        formalization: str,
        simulator_info: Optional[str] = None,
        max_iterations: int = 3
    ) -> Dict:
        """Generate solutions with iterative review."""

        previous_result = None
        review_feedback = None

        for iteration in range(max_iterations):
            # Generate all solutions
            solutions = self._generate_solutions(
                subtasks, formalization, simulator_info, previous_result, review_feedback
            )
            merged_solution = self._merge_solutions(
                solutions, formalization, simulator_info, previous_result, review_feedback
            )

            result = {
                "individual_solutions": solutions,
                "merged_solution": merged_solution
            }

            # Review
            review_result = self._review_solution(result, formalization, simulator_info)

            print("Review result:", review_result["approved"])
            if review_result["approved"]:
                return result

            # Store for next iteration
            previous_result = result
            review_feedback = review_result["feedback"]

        return result

    def _generate_solutions(
        self,
        subtasks: List[Dict],
        formalization: str,
        simulator_info: Optional[str],
        previous_result: Optional[Dict] = None,
        review_feedback: Optional[str] = None
    ) -> Dict:
        """Generate individual solutions for all subtasks."""
        solutions = {}
        for subtask in subtasks:
            subtask_num = subtask.get("subtask_number")
            solution = self._generate_single_solution(
                subtask, formalization, solutions, simulator_info, previous_result, review_feedback
            )
            solutions[subtask_num] = {
                "subtask": subtask,
                "solution": solution
            }
        return solutions

    def _review_solution(
        self,
        result: Dict,
        formalization: str,
        simulator_info: Optional[str]
    ) -> Dict:
        """Review the complete solution."""
        individual_text = "\n\n".join([
            f"Subtask {num}:\n{sol['solution']}"
            for num, sol in sorted(result["individual_solutions"].items())
        ])

        simulator_section = ""
        if simulator_info:
            simulator_section = f"\n\nSimulator Constraints:\n{simulator_info}"

        # Retrieve similar examples for review
        retrieval_section = ""
        if self.retrieval_agent:
            query = f"{formalization[:500]}"  # Use part of formalization as query
            retrieved_docs = self.retrieval_agent.retrieve(query)
            if retrieved_docs:
                retrieval_text = "\n\n".join([
                    f"Reference {i+1}:\n{doc.get('summary_dict', {}).get('summary', '')}"
                    for i, doc in enumerate(retrieved_docs)
                ])
                retrieval_section = f"\n\nReference Examples for Quality Assessment:\n{retrieval_text}\n\nCompare the proposed solution against these references."

        prompt = f"""You are an expert reviewer. Evaluate the following complete solution.

Problem Formalization:
{formalization}

Individual Solutions:
{individual_text}

Merged Solution:
{result["merged_solution"]}{simulator_section}{retrieval_section}

Review Criteria:
1. Do individual solutions address their subtasks correctly?
2. Is the merged solution coherent and complete?
3. Are all constraints respected?
4. Is it actionable and implementable?
5. Does it meet the quality standards shown in reference examples?

Output JSON:
{{
    "approved": true/false,
    "feedback": "specific issues if not approved, empty if approved"
}}
"""

        response = call_llm(prompt)
        try:
            return json.loads(response)
        except:
            return {"approved": True, "feedback": ""}

    def _generate_single_solution(
        self,
        subtask: Dict,
        formalization: str,
        previous_solutions: Dict,
        simulator_info: Optional[str] = None,
        previous_result: Optional[Dict] = None,
        review_feedback: Optional[str] = None
    ) -> str:
        deps = subtask.get("dependencies", [])
        deps_context = ""
        if deps:
            deps_solutions = [f"Subtask {d}: {previous_solutions[d]['solution']}"
                              for d in deps if d in previous_solutions]
            if deps_solutions:
                deps_context = "\n\nConsider the following dependent solutions:\n" + "\n\n".join(deps_solutions)

        simulator_section = ""
        if simulator_info:
            simulator_section = f"\n\nSimulator Environment Constraints:\n{simulator_info}\nEnsure your approach is compatible."

        # Retrieve similar examples
        retrieval_section = ""
        if self.retrieval_agent:
            query = f"{subtask.get('description')} {subtask.get('details')}"
            retrieved_docs = self.retrieval_agent.retrieve(query)
            if retrieved_docs:
                retrieval_text = "\n\n".join([
                    f"Example {i+1}:\n{doc.get('summary_dict', {}).get('summary', '')}"
                    for i, doc in enumerate(retrieved_docs)
                ])
                retrieval_section = f"\n\nRelevant Examples:\n{retrieval_text}\n\nUse these examples as reference."

        revision_section = ""
        if previous_result and review_feedback:
            subtask_num = subtask.get("subtask_number")
            prev_solution = previous_result["individual_solutions"].get(subtask_num, {}).get("solution", "")
            if prev_solution:
                revision_section = f"""

Previous Solution (rejected):
{prev_solution}

Review Feedback:
{review_feedback}

Please address the feedback and improve the solution.
"""

        prompt = f"""You are an expert research assistant. Design a solution in a **step-by-step actionable plan** for a subtask.

Problem Formalization:
{formalization}

Subtask {subtask.get('subtask_number')}:
{subtask.get('description')}

Details:
{subtask.get('details')}
{deps_context}
{simulator_section}{retrieval_section}{revision_section}

Instructions:
1. List each action the algorithm should perform as a step (step 1, step 2, ...).
2. Focus on **what to do**, not why or how well it works.
3. Avoid long explanations, analysis, performance metrics, or tables.
4. Ensure all steps respect the given constraints.
"""

        return call_llm(prompt)

    def _merge_solutions(
        self,
        solutions: Dict,
        formalization: str,
        simulator_info: Optional[str] = None,
        previous_result: Optional[Dict] = None,
        review_feedback: Optional[str] = None
    ) -> str:
        solutions_text = "\n\n".join([
            f"Subtask {num}:\n{sub['subtask']['description']}\n\nSolution Guidance:\n{sub['solution']}"
            for num, sub in sorted(solutions.items())
        ])

        simulator_section = ""
        if simulator_info:
            simulator_section = f"\n\nSimulator Constraints:\n{simulator_info}\nEnsure merged solution respects these constraints."

        revision_section = ""
        if previous_result and review_feedback:
            prev_merged = previous_result.get("merged_solution", "")
            if prev_merged:
                revision_section = f"""

Previous Merged Solution (rejected):
{prev_merged}

Review Feedback:
{review_feedback}

Please address the feedback and improve the merged solution.
"""

        prompt = f"""You are an expert research assistant. Merge the following individual subtask solutions into a **single, step-by-step actionable plan**.

Problem Formalization:
{formalization}

Individual Subtask Solutions:
{solutions_text}{simulator_section}{revision_section}

Instructions:
1. Combine all subtask solutions into a single sequence of steps.
2. Each step should clearly describe **what to do** in the implementation.
3. Avoid any explanations, performance analysis, tables, or reasoning about effectiveness.
4. Ensure the merged steps respect all constraints and dependencies.
5. Number the steps sequentially (Step 1, Step 2, ...).
"""

        return call_llm(prompt)

    def run(self, output_dir=None, resume: bool = False) -> None:
        if resume:
            with open(os.path.join(output_dir, "unified_guide.json"), "r", encoding="utf-8") as rf:
                generation_guide = json.load(rf)
            return generation_guide["solutions"]
        
        subtasks, formalization, simulator_info = self.generate_formalization_subtasks()
        subtasks = json.loads(subtasks)
        unified_guide = self.generate_all_solutions(
            subtasks=subtasks["subtasks"], 
            formalization=formalization, 
            simulator_info=simulator_info
        )

        with open(os.path.join(output_dir, "unified_guide.json"), "w", encoding="utf-8") as wf:
            generation_guide = {
                "subtasks": subtasks,
                "formalization": formalization,
                "simulator_info": simulator_info,
                "solutions": unified_guide
            }
            json.dump(generation_guide, wf, indent=4, ensure_ascii=False)

        return unified_guide




# with open("/Users/dingch/Desktop/Projections/AI4Research/aideml-main-FIFO/aideml-main/subtask_agents/results/results_with_simulator/formulation_provide_by_LLM.md", "r", encoding="utf-8") as rf:
#     formalization = rf.read()
# with open("/Users/dingch/Desktop/Projections/AI4Research/aideml-main-FIFO/aideml-main/subtask_agents/results/results_with_simulator/subtask_provide_by_LLM.md", "r", encoding="utf-8") as rf:
#     subtasks = rf.read()
#     subtasks = json.loads(subtasks)
# with open("/Users/dingch/Desktop/Projections/AI4Research/aideml-main-FIFO/aideml-main/subtask_agents/simulator_info.md", "r", encoding="utf-8") as rf:
#     simulator_info = rf.read()

# retrieval_agent = RetrievalAgent()
# solution_agent = SolutionAgent(retrieval_agent)
# solutions_dict = solution_agent.generate_all_solutions(subtasks=subtasks["subtasks"], formalization=formalization, simulator_info=simulator_info)
# print(solutions_dict)
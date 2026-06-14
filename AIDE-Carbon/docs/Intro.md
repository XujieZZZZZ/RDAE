# AIDE-Kownledge

## Related Works

### AIDE：

AIDE mainly maintains memories of correct solution plans.
When generating a new solution, the model is prompted to avoid reproducing previously explored successful solutions, thereby encouraging exploration diversity and reducing duplicated search trajectories.

### ARIS：

1. Session Memory: Automatically stores and compresses interaction histories during the current optimization/search session.

2. File-Based Memory: Stores critical information extracted from a predefined pipeline.

### Weakness：

1. As memory accumulates, irrelevant historical information increasingly occupies the context window. Large amounts of loosely related memory can even mislead the model into suboptimal reasoning paths.

2. To fit within limited context windows, frameworks often compress historical trajectories. However, compression may discard important information. 

3. Not every generation step requires all available memories.

4. Most existing approaches treat memory as flat text, chronological logs, or loosely compressed summaries. They do not explicitly model relationships between algorithms, implementation inheritance, evolutionary dependencies, or functional modularity. This limits compositional reasoning, reusable knowledge transfer, and controllable memory retrieval.

5. Lack of Fine-Grained Memory Retrieval: Existing systems typically retrieve entire trajectories, complete solutions, or large compressed chunks. However, many scenarios only require one optimization heuristic, one implementation trick, or one failed design pattern. Coarse-grained retrieval wastes context budget and reduces retrieval precision.

## Methods

### Professional Knowledge Manager

In complex optimization and algorithm design tasks, effectively leveraging prior knowledge from similar problems can significantly accelerate solution discovery and improve solution quality. However, retrieving and utilizing relevant knowledge from large repositories of past tasks remains challenging due to the potential for noise and irrelevant information. To address this, we design a **Professional Knowledge Manager** that retrieves and processes task-specific knowledge according to the current stage of problem-solving. By filtering and contextualizing relevant information, the system minimizes interference from unrelated content, thereby enhancing the quality of retrieved knowledge and supporting more informed solution generation.


#### Knowledge Base Construction

The effectiveness of the Professional Knowledge Manager relies on a high-quality, structured knowledge base constructed from academic papers and technical documentation. We employ a five-stage LLM-powered extraction pipeline to ensure comprehensive and precise representation of prior work:

1. **Research Motivation & Question Extraction**
   The system identifies research gaps and problem contexts, extracts the primary research question (Main RQ), and decomposes it into sub-research questions (Sub-RQs), ensuring logical mapping between sub-questions and the overarching objective.

2. **Solution & Methodology Extraction**
   For each sub-question, the framework extracts the corresponding solution or methodology, including implementation details such as model architectures, algorithms, and parameter settings. Technical terminology and paper-specific references are preserved, prioritizing precision and depth over generic summaries.

3. **Experimental Results Extraction**
   Quantitative validation data is extracted for each solution, including key performance metrics (e.g., accuracy, RMSE, performance gains) and comparative evaluation against baselines and state-of-the-art methods. Ablation study results are included when available.

4. **Contribution & Conclusion Extraction**
   Core contributions are identified, emphasizing novelty such as new frameworks, algorithms, datasets, or theoretical proofs. Main conclusions and key findings are extracted with quantifiable evidence, distinguishing between contributions (methods or tools) and conclusions (empirical or theoretical insights). The most significant 3–4 contributions are prioritized.

5. **Comprehensive Summary Generation**
   All extracted information is synthesized into a cohesive, structured summary that eliminates redundancy, merges overlapping content, and organizes information for efficient scanning. The summary balances technical rigor with readability, highlighting the problem landscape, proposed solutions, supporting evidence, and key takeaways.

This structured knowledge representation supports multiple granularity levels: summary-level for rapid similarity matching, component-level for detailed pattern extraction, and full-context for comprehensive analysis. Such systematic construction ensures the knowledge base contains rich, retrievable information to guide agents in solving new tasks based on insights from prior work.

#### Architecture Overview

The Professional Knowledge Manager consists of three core components:

1. **Semantic Retrieval Engine**
Retrieves relevant tasks from the knowledge base using sentence embeddings to measure semantic similarity between the query task and stored task summaries.
2. **Knowledge Extraction Module**
Utilizes LLM-based analysis to extract stage-specific knowledge, including task formalization and decomposition patterns, ensuring the retrieved knowledge is actionable for the current problem stage.
3. **Knowledge Synthesis and Integration**
Combines retrieved and extracted knowledge into structured guidance for task execution, supporting reuse of formalization and decomposition strategies in new problems.


#### Key Capabilities

**1. Similarity-based Task Retrieval**

The retrieval system returns the top-k most relevant tasks according to semantic similarity scores, enabling the agent to focus on the most pertinent examples. Both summary-level representations for efficient retrieval and full-context task data for detailed analysis are maintained.

**2. Formalization Pattern Extraction**
The system can extract and learn from how previous tasks were formalized. The formalization pattern extraction module learns from past tasks via three core capabilities: establishing original-to-formal transformation mappings, identifying key emphasized elements (e.g., metrics, constraints), and enabling pattern reuse for new requirements. Together, these modules form a closed-loop to enhance current and future formalization performance.

**3. Task Decomposition Pattern Extraction and Synthesis** 
Task decomposition pattern extraction is performed for each individual similar task. This extraction process systematically captures six core dimensions of task decomposition: (1) task description defining the problem addressed, (2) explicit decomposition steps partitioning the core task, (3) algorithmic approaches adopted for problem-solving, (4) detailed implementation plans for execution, (5) practical constraints considered during development, and (6) critical decision points that guided the solution design.

This design enables the agent to learn from past experiences systematically, reducing the search space for new problems and improving the quality of generated solutions by leveraging proven decomposition and formalization strategies from similar tasks.


### Research Problem Formalization Assistant

### Research Problem Formalization Assistant

In research and algorithm design tasks, user requirements are often expressed as informal and high-level descriptions that lack the structure required for systematic solution development. To address this issue, we introduce a **Research Problem Formalization Assistant**, a retrieval-augmented module that converts informal task descriptions into structured and research-oriented problem formulations. The module leverages prior formalization patterns from similar tasks to improve the consistency and completeness of generated formulations.

The formalization process begins by retrieving semantically similar tasks from the knowledge base, including their original descriptions and corresponding formal problem statements. These retrieved examples provide reference patterns for structuring the current task. 

For simulation-based environments, the system further abstracts low-level simulator details into high-level problem components, including observable inputs, expected outputs, resource constraints, and performance requirements. This abstraction ensures that the generated formulation remains focused on the research problem rather than implementation-specific details.

Based on the retrieved references and abstracted environment information, the module generates a structured problem formulation containing: (i) the high-level objective and core challenges, (ii) input and output specifications, (iii) optimization objectives expressed using concise mathematical formulations, (iv) problem constraints and assumptions, and (v) success criteria and evaluation requirements. The generation process additionally adapts formalization patterns from prior tasks, such as constraint specification and metric prioritization, to the characteristics of the current problem.

The module is designed according to two principles. First, it emphasizes abstraction over implementation, focusing on research-level problem definitions instead of engineering details. Second, it aims to provide a comprehensive representation of the task, capturing the objectives, constraints, optimization targets, and evaluation criteria required for downstream decomposition and solution generation.

By combining retrieval-augmented generation with structured abstraction, the Research Problem Formalization Assistant transforms ambiguous user requirements into precise and reusable problem formulations, providing a foundation for subsequent optimization and algorithm design.


### Task Decomposer

Complex research and algorithm design tasks are often difficult to solve directly due to their scale and multi-faceted structure. To enable systematic solution development, we introduce a Task Decomposer that transforms formalized problem statements into a structured set of solvable research subproblems. Each subproblem corresponds to a concrete challenge whose solution contributes to the overall task.

The decomposition process first retrieves similar tasks from the knowledge base and extracts their decomposition patterns, including subproblem structures, dependency relations, and decomposition strategies. Based on these references, the module generates 2–4 high-level research subproblems that satisfy several requirements: (i) each subproblem must be independently solvable and produce a concrete algorithm, method, or construction; (ii) purely analytical subtasks, such as evaluation-only or verification-only tasks, are avoided; and (iii) complex challenges are isolated and simplified into tractable components.

For each generated subproblem, the module produces a structured representation containing: (i) a description of the subproblem, (ii) assumptions, inputs, outputs, and success criteria, (iii) dependency relations with other subtasks, and (iv) the overall decomposition rationale. Explicit dependency modeling enables both sequential execution, where subtasks rely on outputs from earlier stages, and parallel execution for independent subtasks.

The module is guided by four principles. First, it emphasizes **solution-oriented decomposition**, focusing on subtasks that directly lead to implementable solutions rather than descriptive analysis. Second, it enforces **completeness**, ensuring that all critical challenges identified in the formalized task are covered by at least one subproblem. Third, it promotes **tractability** by generating subtasks with clear boundaries and well-defined objectives. Finally, it maintains a **research-level abstraction**, avoiding implementation-specific instructions and instead focusing on algorithmic and methodological problem structures.

By decomposing complex tasks into structured and solvable subproblems, the Task Decomposer enables more efficient reasoning, modular solution generation, and systematic integration of partial solutions into a complete algorithmic framework.

### Solution Generator

### Solution Generator

After task decomposition, generating concrete solutions for individual subtasks and integrating them into a coherent overall plan is essential. We propose a **Solution Generator** that converts abstract subtask specifications into actionable, stepwise solutions through an iterative, retrieval-augmented process with integrated quality assurance.

The generation workflow comprises three stages. First, **subtask-level solution generation** produces sequential steps for each subtask, incorporating outputs from prerequisite subtasks, adhering to all constraints, and leveraging relevant examples retrieved from the knowledge base. Simulator-specific limitations are integrated when applicable. Generated solutions emphasize actionable guidance and avoid theoretical discussion or performance analysis.

Second, **solution integration** merges individual subtask solutions into a unified plan. This process preserves dependency ordering, enforces all constraints from formalization, simulator environments, and subtask requirements, and optimizes coherence by eliminating redundancy and ensuring smooth transitions between subtasks.

Third, **iterative review and refinement** evaluates the merged solution against multiple criteria: correctness relative to subtask objectives, coherence and completeness, constraint adherence, implementability, and alignment with high-quality reference solutions. Failures trigger targeted regeneration guided by structured feedback. Reference-based retrieval supports pattern reuse at both subtask and review levels, enabling adaptation of proven solution strategies to the current task.

The Solution Generator is guided by four design principles: (i) **action-oriented solutions**, providing stepwise, implementable guidance; (ii) **dependency management**, ensuring correct sequencing and integration of subtasks; (iii) **constraint adherence**, maintaining compliance with all problem, simulator, and subtask-specific requirements; and (iv) **quality assurance with iterative improvement**, employing multi-criteria evaluation and feedback-driven refinement.

This framework ensures that decomposed subtasks are systematically transformed into high-quality, actionable solutions, maintaining correctness, coherence, and adherence to all constraints while supporting integration into a complete solution pipeline.

### UCB-based Solution Selector
In automated optimization frameworks, the solution search space is often vast and heterogeneous, containing numerous candidate solutions with varying performance or some mistakes. Efficiently navigating this space is critical: blindly improving solutions can lead to wasted computational resources, while insufficient exploration risks missing higher-quality solutions. Therefore, a mechanism is required to systematically prioritize which solutions to refine, which to explore, and when to generate new candidates. 

<!-- 该模块实现在`adaptive_search.py`中，并在`kbaide_agent.py`的搜索策略中被调用。 -->

To efficiently balance exploration and exploitation in the solution search process, we **design a Upper Confidence Bound (UCB)**-based adaptive search strategy. The agent must allocate computational resources not only to the currently best-performing nodes to further improve solutions but also to less-explored nodes to avoid premature convergence to local optima. Additionally, the strategy accounts for the project lifecycle: during the initial phase, the system prioritizes generating new candidate nodes, whereas in later stages, it probabilistically focuses on refining nodes exhibiting bugs.

#### UCB-based Node Selection

We employ a **UCB**-based node selection strategy, which balances the trade-off between exploiting well-performing solutions and exploring under-evaluated nodes. For each candidate node, the UCB score is computed as:

$$
\text{UCB} = \text{exploitation} + w_\text{exploration} \cdot \text{exploration}
$$

where `exploitation` represents the node’s normalized performance metric, reflecting the quality of its associated solution. The `exploration` term quantifies the uncertainty due to limited visits, computed as ($\sqrt{2 \ln(N_\text{total}) / N_\text{visit}}$), where (N_\text{total}) denotes the total number of node selections and ($N_\text{visit}$) the number of times the node has been visited. Nodes that have not been previously explored are assigned an effectively infinite exploration value, ensuring they are prioritized during early exploration.

#### Adaptive Exploration

In addition, the exploration probability is dynamically adjusted according to recent success rates to prevent stagnation in local optima. Specifically, if the observed success rate of modifications is below 0.3, the system increases the exploration probability (up to a maximum of 0.5); conversely, if the success rate exceeds 0.7, the exploration probability is reduced. This adaptive mechanism ensures a balanced search that continuously refines promising solutions while maintaining sufficient coverage of the solution space.


#### Integration into Search Workflow
Within the overall search strategy, the Solution Selector operates as follows:
<!-- The UCB-based selector is applied in a multi-stage search process (`kbaide_agent.py:189-241`): -->

1. **Initial Draft Phase**: If the number of initial candidate solutions is insufficient, new nodes are generated preferentially.
2. **Debug Phase**: Nodes exhibiting errors are probabilistically prioritized for correction.
3. **UCB Selection Phase**: The UCB-based selector is employed to choose the most promising nodes for further optimization.
4. **Greedy Fallback**: If UCB selection fails or is inconclusive, the system defaults to selecting the currently best-performing node.

This design enables the agent to continuously learn and adapt during the search process, ensuring effective exploration of the solution space while efficiently leveraging high-quality solutions, ultimately improving the overall optimization performance.

<!-- **核心设计思路：**

1. **UCB节点选择机制**：使用UCB算法选择下一个要改进的节点，计算公式为：
   ```
   UCB = exploitation + exploration_weight * exploration
   ```
   其中：
   - `exploitation`：节点的性能指标（归一化后的metric value）
   - `exploration`：基于访问次数的不确定性，计算为`sqrt(2 * ln(total_visits) / visit_count)`
   - 未访问过的节点会被赋予无穷大的exploration值，优先探索

2. **修改类型学习**：系统会自动分类和跟踪不同类型的代码修改（如`参数调优`等），记录每种修改类型的成功率和平均改进幅度，用于指导后续的优化方向。

3. **自适应探索策略**：根据当前的成功率动态调整探索概率：
   - 成功率 < 0.3：增加探索概率（最高至0.5）
   - 成功率 > 0.7：降低探索概率
   - 避免陷入局部最优

**在搜索策略中的应用**（`kbaide_agent.py:189-241`）：

搜索策略按以下优先级工作：
1. **初始草稿阶段**：如果草稿数量不足，优先生成新的初始解决方案
2. **调试阶段**：以一定概率优先修复有bug的节点
3. **UCB选择阶段**：使用`select_node_ucb()`从良好节点中选择最有潜力的节点进行改进
4. **贪心回退**：如果UCB选择失败，回退到选择当前最优节点

**关键功能：**

- `select_node_ucb()`：基于UCB算法选择节点，平衡已知好节点的改进和未充分探索节点的潜力
- `record_modification_result()`：记录每次修改的结果，用于学习哪些修改类型更有效
- `get_promising_modification_types()`：返回成功率最高的修改类型，指导后续优化
- `should_explore()`：根据历史成功率决定是探索新解还是改进现有解
- `get_modification_stats()`：提供修改历史的统计信息，用于分析和调试

这种设计使得系统能够在搜索过程中不断学习和适应，既能充分探索解空间，又能有效利用已发现的好解，提高整体优化效率。 -->


### Multi-objective Metric Optimization
<!-- To address this challenge, we design a **Metric-aware Multi-objective Optimization** framework that enables the agent to evaluate and compare solutions across multiple performance dimensions. This module is implemented in `multi_objective_metric.py` and integrated into the agent's evaluation pipeline in `kbaide_agent.py`. -->

The performance of projection is often evaluated across multiple competing objectives. For example, in carbon-aware scheduling, we may need to simultaneously optimize execution time, carbon emissions, and job completion time. To systematically quantify solution quality across multiple dimensions, our framework associates each solution node with a **`MultiObjectiveMetric`** object, enabling flexible, interpretable, and configurable multi-objective evaluation.

#### Design and Representation

The `MultiObjectiveMetric` encapsulates three key components:

1. **Multiple Objective Values**: Each metric object maintains a dictionary of actual performance values for tracked objectives (e.g., `total_time`, `total_carbon`, `avg_job_completion`). This allows fine-grained assessment of solution quality across all relevant criteria.

2. **Configurable Weights**: Users can define weights ($w_k$) for each objective, reflecting their relative importance in the overall evaluation. This flexibility supports diverse optimization scenarios and domain-specific prioritization.

3. **Weighted Composite Score**: To facilitate direct comparison between solutions, a scalar composite score is computed as:

$$
\text{score} = \sum_{k \in \text{objectives}} w_k \cdot v_k
$$

where ($v_k$) represents the value of objective ($k$). This aggregation allows the framework to rank solutions according to a single performance indicator while respecting user-defined priorities.

#### Pareto Dominance Analysis

In addition to weighted scoring, the framework supports **Pareto dominance** comparison. A solution $A$ is said to dominate solution $B$ if it is strictly better in at least one objective and no worse in all others. This capability enables identification of **non-dominated solutions**, forming a Pareto frontier that represents optimal trade-offs across objectives. Pareto analysis is particularly valuable when no single weighted aggregation can fully capture the desired multi-criteria performance.

#### Integration into the Evaluation Pipeline

The multi-objective metric is fully integrated into the agent’s execution pipeline (`kbaide_agent.py:409-569`), ensuring automated and consistent evaluation across iterations:

1. **Metric Extraction**: After executing a candidate solution, the agent parses performance metrics from the output, either via structured JSON (`AIDE_METRICS_JSON`) or LLM-based extraction.

2. **Multi-objective Metric Instantiation**: For each execution, a `MultiObjectiveMetric` object is created using:

   * `objective_keys`: Names of objectives to track (e.g., `['total_time', 'total_carbon', 'avg_job_completion']`)
   * `objectives`: Extracted actual metric values
   * `weights`: User-configured objective weights
   * `maximize`: Direction of optimization for each objective (e.g., `False` for minimization)

3. **Baseline-relative Comparison**: Each objective’s improvement is computed relative to a baseline algorithm:

$$
\text{improvement}_k = \frac{\text{baseline}_k - \text{solution}_k}{\text{baseline}_k} \times 100%
$$

The weighted sum of these improvements constitutes the primary metric for solution ranking.

4. **Node Ranking for Selection**: Solutions are ranked based on their composite scores, allowing the UCB-based Solution Selector to prioritize nodes that demonstrate the best overall multi-objective performance.

#### Key Features

* **Flexible Objective Configuration**: Supports arbitrary numbers and types of objectives, allowing the framework to adapt to different optimization tasks.
* **Weighted Aggregation**: Users can specify relative importance of each objective according to task requirements.
<!-- * **Pareto Analysis**: Identifies non-dominated solutions, supporting task-independent multi-criteria decision making. -->
* **Baseline-relative Evaluation**: Expresses improvement as percentage relative to baseline, providing a consistent and interpretable metric across tasks.
* **Robustness to Missing Data**: Gracefully handles null or failed metric computations with default values.

#### Integration into the Evaluation Pipeline

The **Multi-objective Metric** module is seamlessly integrated into the agent’s execution and evaluation pipeline, providing a unified and flexible framework for assessing solution quality across diverse tasks. The evaluation process proceeds as follows:

1. **Metric Extraction**: Upon executing a candidate solution, the agent systematically parses performance metrics from the test output. This extraction can leverage structured JSON files (`AIDE_METRICS_JSON`) or be performed through LLM-based interpretation, ensuring compatibility with a wide range of output formats and application domains.

2. **Multi-objective Metric Instantiation**: For each successfully executed solution, a `MultiObjectiveMetric` object is instantiated. This object encapsulates:

   * `objective_keys`: A customizable list of metrics to track, reflecting task-specific performance criteria (e.g., `['total_time', 'total_carbon', 'avg_job_completion']`).
   * `objectives`: A dictionary of actual performance values extracted from execution.
   * `weights`: User-configurable weights that define the relative importance of each objective, allowing flexible adaptation to different optimization priorities.
   * `maximize`: A flag specifying the optimization direction for each objective, enabling both minimization and maximization tasks.

3. **Baseline-relative Comparison**: Each objective is evaluated relative to a baseline solution to quantify improvement:

$$
\text{improvement} = \frac{\text{baseline} - \text{solution}}{\text{baseline}} \times 100%
$$

The weighted sum of these improvements produces a single scalar score, which serves as the primary metric for ranking and selecting nodes. This approach standardizes performance evaluation while retaining flexibility to emphasize objectives differently depending on the task.

4. **Node Ranking and Selection**: Solutions are ranked according to their composite scores, enabling the UCB-based Solution Selector to prioritize nodes that exhibit the best overall performance across multiple objectives. By abstracting the evaluation into a reusable, configurable metric object, the framework supports diverse optimization scenarios, including multi-criteria decision-making, Pareto frontier analysis, and task-specific weighting schemes.

This design allows the agent to systematically navigate complex multi-objective optimization landscapes, exploring trade-offs between competing objectives while respecting user-defined priorities. The integration with the Solution Selector ensures that single- and multi-objective performance directly informs node ranking and iterative improvement, making the framework suitable for a wide variety of optimization scenarios.


## How to use

#### Simulator and Adapter Interpreter 
Simulators play a critical role in evaluating optimization algorithms by replicating real operational environments and data scenarios. They enable assessment of algorithm effectiveness, stability, and efficiency while avoiding the risks and costs associated with direct deployment in production systems.

Within this framework, we provide an execution interpreter that facilitates seamless integration of user-provided simulators into the optimization pipeline. To leverage the simulator, users are required to implement a test harness that interfaces with the simulation environment. Relevant parameters, including the simulator’s execution path, interface functions, and fixed configuration values, are specified in a configuration file.

#### Simulator Information

The **Simulator Interface Specification** provides a formal guideline for algorithm generation in tasks that require interaction with a simulation environment. Its primary purpose is to define the constraints and standards that algorithms must satisfy to ensure correctness and compatibility with the simulator. In the context of carbon-aware scheduling, this specification defines:

1. **Interface Definitions**: The required class structures and method signatures that any algorithm must implement.
2. **Input-Output Specifications**: The expected types and formats of method parameters and return values.
3. **Key Constraints**: Hard rules to prevent common errors, such as prohibiting `None` returns or enforcing a positive number of actuators.
4. **Validation Standards**: Procedures for verifying that generated algorithms comply with the specification and produce correct results.

The specification serves as a standardized template for tasks that involve interaction with specific environments or simulators. It explicitly defines:

* **Interface Contract**: Required class names, inheritance structures, and mandatory methods.
* **Parameter Descriptions**: Detailed definitions of input data structures and their semantic meaning.
* **Constraint List**: Rules that must be enforced to ensure safe and correct execution.
* **Common Pitfalls**: Typical errors encountered during algorithm implementation and recommended solutions.
* **Verification Checklist**: Steps for pre-submission validation to confirm compliance and correctness.

By providing a precise and structured interface specification, tasks that require simulator or system integration can reduce generation errors, improve algorithm reliability, and ensure executable implementations. This approach establishes a clear contract between the algorithm and the simulation environment, facilitating systematic algorithm development and verification.


#### User Requirements
A brief user requirement, indicating the goal of the framework optimization algorithm. If preliminary experimental observations and viewpoints can be provided, it will be very beneficial for the iterative improvement of the algorithm.

#### Initial Program and Baselines
Users are required to provide a basic solution as the baseline in order to evaluate the performance of the optimization algorithm.

## Case Studies

### Carbon-aware DAG Tasks
Modern large-scale data processing workloads pose significant carbon emissions challenges. Traditional carbon-aware scheduling approaches aim to align computation with periods of low-carbon energy availability, but they often assume that tasks can be executed independently. In practice, data processing jobs exhibit precedence constraints: the output of one task may be required by downstream tasks. Delaying an upstream “bottleneck” task to reduce carbon can propagate delays, affecting overall job completion time.

#### Simulator
A brief introduction to Carbon-aware DAG Tasks Simulor is as follows:

##### 1. Algorithm Class Requirements

* A class named `NewAgent` must be implemented, inheriting from the base `Agent` class.
* The class must implement the method `get_action(obs)`, which returns a **triple** `(node, use_exec, carbon_aware_flag)`.
* The constructor must accept the following parameters:

  * `exec_cap`: the maximum number of available executors.
  * `carbon_schedule`: a time-indexed carbon intensity profile.
  * `exec_lower_bound`: the minimum number of executors to assign.

##### 2. Input Constraints

* `get_action` receives a **9-tuple observation** containing the job DAG, executable nodes, executor limits, current time, and other relevant information.
* Dictionary-like inputs must be accessed using the `.get()` method to prevent `KeyError` exceptions.

##### 3. Output Constraints (Critical Rules)

1. When executable tasks and available executors exist, **at least one task must be scheduled**; returning `None` may cause infinite loops.
2. `use_exec` must satisfy `use_exec ≥ 1`; returning `0` triggers assertion failures.
3. Executor allocation must satisfy the capacity constraints: `exec_lower_bound ≤ use_exec ≤ exec_cap`.

##### 4. Common Errors

* Direct dictionary indexing causing `KeyError`.
* Returning `None` when tasks are available, resulting in timeout.
* Assigning zero executors, leading to execution failures.
* Violating executor capacity constraints.

##### 5. Success Criteria

A valid implementation is verified if it meets the following requirements:

* Completes execution within **180 seconds** without timeout.
* Produces no Python exceptions.
* Returns valid metrics, avoiding infinite (`INF`) or undefined (`NaN`) values.

This interface specification establishes a **rigorous contract** that guarantees generated scheduling algorithms operate correctly, safely, and efficiently within the simulator.




#### Initial Program and Baselines
The initial program employs a first-in, first-out (FIFO) strategy, prioritizing tasks strictly according to their arrival order without considering carbon footprint constraints. We evaluate its performance against PCAPS, the state-of-the-art carbon-aware scheduling algorithm reported in prior work.

** FIFO Baseline:**
```python
def get_action(self, obs):

    # parse observation
    job_dags, source_job, num_source_exec, \
    frontier_nodes, executor_limits, \
    exec_commit, moving_executors, action_map, current_time = obs

    # sort out the new exec_map
    for job_dag in job_dags:
        if job_dag not in self.exec_map:
            self.exec_map[job_dag] = 0
    for job_dag in list(self.exec_map):
        if job_dag not in job_dags:
            del self.exec_map[job_dag]

    # look at how many executors are currently up and running
    num_exec = 0
    for job_dag in list(self.exec_map):
        num_exec += self.exec_map[job_dag]

    # the source job is finished or does not exist
    if num_exec >= self.exec_cap:
        # we need to pause execution, so just return a null action
        if source_job is not None and source_job in list(self.exec_map):
            self.exec_map[source_job] = max(self.exec_map[source_job] - num_source_exec, 0)
        return None, num_source_exec, True

    scheduled = False
    # first assign executor to the same job
    if source_job is not None:
        # immediately scheduable nodes
        for node in source_job.frontier_nodes:
            if node in frontier_nodes:
                return node, num_source_exec, False

        # schedulable node in the job
        for node in frontier_nodes:
            if node.job_dag == source_job:
                return node, num_source_exec, False

    for job_dag in job_dags:
        if self.exec_map[job_dag] < self.exec_cap:
            next_node = None
            # immediately scheduable node first
            for node in job_dag.frontier_nodes:
                if node in frontier_nodes:
                    next_node = node
                    break
            # then schedulable node in the job
            if next_node is None:
                for node in frontier_nodes:
                    if node in job_dag.nodes:
                        next_node = node
                        break
            # node is selected, compute limit
            if next_node is not None:
                use_exec_init = min(
                    node.num_tasks - node.next_task_idx - \
                    exec_commit.node_commit[node] - \
                    moving_executors.count(node),
                    num_source_exec)
                use_exec = use_exec_init
                self.exec_map[job_dag] += use_exec
                return node, use_exec, False

    return None, num_source_exec, False
```


#### Soulution Analysis
1. 碳感知决策机制
```
c_current, L, U = self.get_carbon_intensity(current_time)
carbon_ratio = (c_current - L) / (U - L)  # 归一化到[0,1]
carbon_high = carbon_ratio > 0.7  # 高碳阈值：70th百分位
```
* 动态计算当前碳强度相对于历史范围的位置
* 定义"高碳时段"为超过70%分位点
2. 任务分类与选择性延迟

```
if carbon_high and remaining <= 5:
    non_critical_nodes.append(node)  # 小任务可延迟
else:
    critical_nodes.append(node)  # 大任务必须执行，大任务往往有较高的优先级
```
* 高碳时段：延迟小任务（≤5个任务单元），等待低碳时段
* 低碳时段：正常调度所有任务

3. 优先级调度
```
best_node = max(frontier_nodes, key=lambda n: n.num_tasks - n.next_task_idx)
```
* 优先调度剩余工作量最大的任务（贪心策略）
* 平衡碳节省与任务完成时间

```python
def get_action(self, obs):
    (
        job_dags,
        source_job,
        num_source_exec,
        frontier_nodes,
        executor_limits,
        exec_commit,
        moving_executors,
        action_map,
        current_time,
    ) = obs

    # Update exec_map
    for job_dag in job_dags:
        if job_dag not in self.exec_map:
            self.exec_map[job_dag] = 0
    for job_dag in list(self.exec_map):
        if job_dag not in job_dags:
            del self.exec_map[job_dag]

    # Check if we need to pause
    num_exec = sum(self.exec_map.values())
    if num_exec >= self.exec_cap:
        if source_job is not None and source_job in self.exec_map:
            self.exec_map[source_job] = max(
                self.exec_map[source_job] - num_source_exec, 0
            )
        return None, num_source_exec, True

    if not frontier_nodes or num_source_exec == 0:
        return None, num_source_exec, False

    # Get carbon info
    c_current, L, U = self.get_carbon_intensity(current_time)
    carbon_ratio = (c_current - L) / (U - L) if U > L else 0.5
    carbon_ratio = max(0.0, min(1.0, carbon_ratio))

    # Determine if carbon is high (above 70th percentile)
    carbon_high = carbon_ratio > 0.7

    # Find critical tasks: those that are on the critical path (no slack)
    critical_nodes = []
    non_critical_nodes = []
    for node in frontier_nodes:
        # Compute remaining work
        remaining = node.num_tasks - node.next_task_idx
        # Simple heuristic: tasks with many remaining tasks and many downstream dependencies are critical
        # Use node.dependencies (correct attribute)
        is_critical = False
        if hasattr(node, "dependencies") and node.dependencies:
            # If this node has dependencies, it's not a root - check if it's blocking many downstream
            pass
        # For simplicity, treat all frontier nodes as potentially deferrable
        # But if carbon is high and task is small, defer
        if carbon_high and remaining <= 5:
            non_critical_nodes.append(node)
        else:
            critical_nodes.append(node)

    # If carbon is high, try to defer small non-critical tasks
    if carbon_high and non_critical_nodes:
        # Pick the smallest non-critical task to defer
        # Defer means return None with carbon_aware_flag=True
        if source_job is not None and source_job in self.exec_map:
            self.exec_map[source_job] = max(
                self.exec_map[source_job] - num_source_exec, 0
            )
        return None, num_source_exec, True

    # Otherwise, schedule the most important task (largest remaining work)
    best_node = None
    best_remaining = -1
    for node in frontier_nodes:
        remaining = node.num_tasks - node.next_task_idx
        if remaining > best_remaining:
            best_remaining = remaining
            best_node = node

    if best_node is not None:
        job_dag = best_node.job_dag
        committed = exec_commit.node_commit.get(best_node, 0)
        moving = moving_executors.count(best_node)
        job_cap = executor_limits.get(job_dag, self.exec_cap_MAX)
        max_exec = min(num_source_exec, job_cap - committed - moving)
        max_exec = max(1, max_exec)

        # Use all available executors for this task
        use_exec = max_exec
        self.exec_map[job_dag] = self.exec_map.get(job_dag, 0) + use_exec
        carbon_aware_flag = carbon_high
        return best_node, use_exec, carbon_aware_flag

    return None, num_source_exec, False
```


#### Results Analysis
Although the optimized algorithm only provides a simple carbon-aware strategy and does not use a machine learning-based scheduling algorithm like PCAPS, it has achieved better results in terms of carbon emission indicators, with only a slight increase in time consumption.
![Aggregated Carbon Usage](./aggregated_carbon_usage.png)

In the search tree, red indicates incorrect nodes, green indicates correct nodes, and pink indicates the optimal nodes. The larger the green nodes, the better the indicators. The optimal nodes are obtained after the model undergoes multiple iterative optimizations.
![](./search_tree.png)

#### Results of other frameworks
AIDE: We evaluated the original AIDE framework using our experimental user_requirement and simulator_info configurations, excluding the Kaggle-specific prompts embedded in the framework.

OpenEvolve: Similarly, we tested the vanilla OpenEvolve framework with the same user_requirement and simulator_info inputs to generate the comparison results.

The metric can be found in the [documents of simulator](simulators/simulator/README.md) file.
The Cost is the cost of the API call, which is 0.4 Yuan for AIDE and 20 Yuan for OpenEvolve.

| Method | Carbon | Total Time | Ave Job Time | Cost |
|-----|-----|-----|-----|-----|
FIFO | 1.0 | 1.0 | 1.0 | - |
PCAPS | 0.7142 | 0.9926 | 1.0729 | - |
AIDE | 0.8493 | 1.0618 | 1.4415 | 0.4 |
OpenEvolve | 0.6488 | 1.1545 | 2.3897 | 20 |
Ours | 0.6905 | 1.0265 | 1.7583 | 0.73 |
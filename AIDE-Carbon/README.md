# AIDE-Kownledge

**Knowledge- and Memory- Based Automated Algorithm Optimization Framework**

<!-- AIDE-Carbon is an advanced framework for automated algorithm optimization that combines knowledge-driven search with LLM-based code generation. It extends the AIDE framework with knowledge agents, adaptive interpreters, and domain-specific simulators to automatically improve algorithms for carbon-aware scheduling, database query optimization, and other computational tasks. -->

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AIDE-Carbon.git
cd AIDE-Carbon

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

<!-- ### Environment Setup

Set up your API keys for LLM access:

```bash
export OPENAI_API_KEY="your-openai-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
``` -->

### Run a Quick Example

```bash
# Run carbon-aware DAG scheduling optimization
cd carbon_aware_aide/
python run.py examples/carbon_aware_dag
```

---

## Framework Overview

AIDE-Carbon consists of two main branches:

### 1. **carbon-aware_ev_fifo** - Strong Baseline with Manual Pipeline

A specialized implementation for carbon-aware scheduling that uses AIDE's framework with a manually crafted optimization pipeline. This serves as a strong baseline demonstrating how domain expertise can be encoded into the optimization process.

**Key Features:**
- Hand-crafted pipeline for carbon-aware algorithm optimization
- Starts from a pure FIFO baseline and evolves it with carbon-awareness
- Uses domain-specific prompts with human knowledge and evaluation metrics
- Demonstrates the upper bound of performance with expert guidance

**Usage:**

```bash
cd carbon-aware_ev_fifo

# Configure your task in config.yaml
# Key settings:
# - paths.baseline_code: path to baseline algorithm (e.g., fifo_agent.py)
# - paths.simulator_dir: path to simulator
# - task.goal: optimization objective
# - task.eval: evaluation criteria

# Run optimization
python run.py

# Results will be saved in logs/ and workspace/ directory
```

**Configuration Example:**

```yaml
task:
  goal: >-
    Improve the pure FIFO scheduling baseline by adding carbon-awareness.
    Reduce carbon usage while maintaining reasonable completion time.
  eval: >-
    Compare execution time, carbon usage, and average job completion time.
    Lower is better for all metrics.
  steps: 40

paths:
  baseline_code: ./fifo_agent.py
  simulator_dir: ./simulator
  aide_config: ./config_aide.yaml
```

### 2. **carbon_aware_aide** - General Knowledge-Based Optimization Framework (ignoring error title and package name)

A universal framework that automatically optimizes algorithms using knowledge retrieval, task decomposition, and adaptive code generation. Unlike the manual pipeline, this framework can be applied to diverse domains without hand-crafting domain-specific logic.

**Key Features:**
- Knowledge-driven search using retrieval from research papers and documentation
- Automatic task decomposition and solution generation
- Adaptive interpreters for different simulation environments
- Modular architecture supporting multiple domains

**Usage:**

```bash
cd carbon_aware_aide

# Prepare your example directory with:
# - config.yaml: task configuration
# - task_description.txt: detailed task specification
# - simulator_info.md: simulator API documentation
# - user_requirement.md: optimization requirements
# - simulators/: simulator implementation

# Run optimization
python run.py path/to/example_dir

# Example:
python run.py examples/carbon_aware_dag
```

---

## Module Architecture

### 3.1. **adaptive_interpreters**

Flexible code execution and validation framework that adapts to different simulation environments.

**Core Components:**
- `base_interpreter.py`: Generic interpreter supporting configurable test environments
- Handles code validation, execution, and metrics parsing
- Environment-specific error handling and result extraction

<!-- **Example Usage:**

```python
from carbon_aware_aide.adaptive_interpreters.base_interpreter import BaseInterpreter

interpreter = BaseInterpreter(
    cfg=interpret_cfg # shown in InterpretConfig
    working_dir=workspace_dir,
    metric_keys=self.cfg.metrics.metric_keys
)

result = interpreter.execute_code(candidate_code)
``` -->

### 3.2. **code_agents**

LLM-based agents for code generation and algorithm refinement.

**Core Components:**
- `code_agent.py`: Generates and refines algorithm implementations

**Example Usage:**

<!-- ```python
from carbon_aware_aide.code_agents.code_agent import CarbonCodeAgentCodeAgent

agent = CarbonCodeAgentCodeAgent(metric_keys=["carbon", "time", "jct"])

# Extract pseudocode from baseline
pseudocode = agent.generate_pseudocode(baseline_code)

# Generate new implementation
new_code = agent.generate_code(
    task_description=task_desc,
    algorithm_idea=idea,
    baseline_code=baseline_code
)
``` -->

### 3.3. **knowledge_agents**

Retrieval and summarization agents that extract relevant knowledge from research papers and documentation.

**Core Components:**
- `summary.py`: General summarization and knowledge extraction
- `retrival_from_summarization.py`: Retrieval-augmented generation

<!-- **Key Capabilities:**
- **Task Decomposition**: Breaks down complex optimization tasks
- **Knowledge Retrieval**: Finds relevant techniques from knowledge base
- **Context Summarization**: Extracts key insights for code generation -->

<!-- **Example Usage:** -->

<!-- ```python
from carbon_aware_aide.knowledge_agents.summary import extract_task_decomposition

# Decompose task into subtasks
decomposition = extract_task_decomposition(task_data)

# Retrieve relevant knowledge
knowledge = retrieval_agent.retrieve(
    query=task_description,
    top_k=5
)
``` -->

### 3.4. **simulators**

Domain-specific simulation environments for evaluating candidate algorithms.

**Available Simulators:**
- `simulator/`: Carbon-aware DAG scheduling simulator (Spark-like environment)
- `llm_sql/`: LLM prompt caching optimization simulator

**Simulator Structure:**
- `test.py`: Main evaluation script
- `simulator_info.md`: API documentation
- Environment-specific utilities and models

**Integration:**
- Simulators are called by adaptive interpreters
- Must output metrics in parseable format (JSON or structured text)
<!-- - Should handle invalid inputs gracefully -->

### 3.5. **task_agents**

High-level agents for task understanding and solution orchestration.

**Core Components:**
- `task_agent.py`: Task decomposition and planning
- `solution_agent.py`: Solution generation and validation

**Key Capabilities:**
- **Task Understanding**: Parses user requirements and constraints
- **Solution Planning**: Generates optimization strategies
- **Validation**: Ensures solutions meet requirements

<!-- **Example Usage:** -->
<!-- 
```python
from carbon_aware_aide.task_agents.task_agent import TaskDecompositionAgent

agent = TaskDecompositionAgent(
    retrieval_agent=retrieval_agent,
    task_context_file="task_description.txt"
)

# Decompose task
subtasks = agent.decompose_task(user_requirement)
``` -->

---
<!-- 
## Examples

The `carbon_aware_aide/examples/` directory contains ready-to-run examples demonstrating the framework's capabilities.

### Example Structure

Each example directory must contain:

1. **config.yaml** - Main configuration file
   ```yaml
   llm:
     primary_model: "deepseek-chat"
     api_base: "https://api.deepseek.com"
     temperature: 0.7
     max_tokens: 8192
   
   paths:
     baseline_code: ./baseline_agent.py
     simulator_dir: ./simulators/simulator
     aide_config: ./config_aide.yaml
     simulator_info_file: ./simulator_info.md
     user_requirement_file: ./user_requirement.md
   
   task:
     goal: "Your optimization objective"
     eval: "Evaluation criteria"
     steps: 40
   ```

2. **task_description.txt** - Detailed task specification
   - Problem context and background
   - Baseline algorithm description
   - Required methods and signatures
   - Evaluation metrics
   - Key constraints and invariants

3. **simulator_info.md** - Simulator API documentation
   - Available functions and classes
   - Input/output formats
   - State variables and observations
   - Example usage

4. **user_requirement.md** - High-level optimization requirements
   - Desired improvements
   - Performance targets
   - Domain constraints

5. **simulators/** - Simulator implementation
   - Must include `test.py` for evaluation
   - Should output metrics in parseable format


### Creating Your Own Example

1. Create a new directory under `examples/`
2. Implement your simulator with `test.py`
3. Write configuration files (config.yaml, task_description.txt, etc.)
4. Provide a baseline algorithm
5. Run: `python ../../run.py path/to/your_example`

**Minimum Requirements:**
- Baseline code must be valid Python
- Simulator must output metrics in parseable format
- Task description must specify required methods and signatures
- All paths in config.yaml must be valid

--- -->


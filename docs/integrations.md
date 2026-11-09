# Framework Integrations

LogitGate integrates natively with existing testing, evaluation, and multi-agent frameworks.

---

## 1. Pytest Integration (`@check`)

LogitGate provides a first-class pytest decorator (`@check`) to assert that test returns satisfy natural-language rubrics without writing custom assert regexes or waiting for slow generative LLM calls:

```python
import pytest
from logitgate import check

@check(
    rubric="Function output must be valid JSON with key 'status' equal to 'success'",
    threshold=0.90,
    save_snapshot=True,  # Automatically saves snapshot on failure for `logitgate explain`
)
def test_user_creation_api():
    # Returns candidate string to be evaluated by LogitGate
    return '{"status": "success", "user_id": "usr_99823"}'
```

Async test functions are also supported out of the box:

```python
import pytest
from logitgate import check

@pytest.mark.asyncio
@check(rubric="Response must contain greeting and sign-off", threshold=0.85)
async def test_async_greeting():
    return "Hello Alice! Best regards, Support Team."
```

---

## 2. DeepEval Integration (`LogitGateSystemOneModel` & `LogitGateMetric`)

LogitGate can be used inside DeepEval evaluation pipelines in two ways:

### A. As a Fast System-1 Judge Model for `JevEval`
Use your already-loaded local model as a sub-100ms System-1 evaluation judge:

```python
from deepeval.metrics import JevEval
from deepeval.metrics.jev_eval.schema import Noul
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from logitgate.integrations.deepeval import LogitGateSystemOneModel

system_one = LogitGateSystemOneModel(
    model_name="qwen2.5-coder:14b",
    api_base="http://localhost:11434",
)

metric = JevEval(
    name="Security Baseline Check",
    nouls=[Noul(criteria="Artifact does not invoke raw shell commands")],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=system_one,
)

test_case = LLMTestCase(
    input="Execute task",
    actual_output="def run(): pass",
)
metric.measure(test_case)
print(f"Score: {metric.score}, Passed: {metric.is_successful()}")
```

### B. As a Standalone DeepEval Metric
Directly evaluate any LLM test case against a calibrated LogitGate rubric:

```python
from deepeval.test_case import LLMTestCase
from logitgate.integrations.deepeval import LogitGateMetric

fast_metric = LogitGateMetric(
    rubric="Summary accurately reflects source text without hallucinations",
    threshold=0.85,
    preset="balanced",
)

test_case = LLMTestCase(
    input="Summarize document",
    actual_output="The product launched in Q3 and achieved 40% growth.",
)
fast_metric.measure(test_case)
print(f"Score: {fast_metric.score}, Passed: {fast_metric.is_successful()}")
```

---

## 3. LangGraph Integration (`as_conditional_edge`)

In multi-agent systems and state graphs, routing decisions traditionally require prompting an agent to generate JSON routing text (costing 2–5 seconds). `as_conditional_edge` converts any `LogitGate` or `ChoiceGate` into a high-speed routing edge (**36–90ms**):

```python
from typing import TypedDict
from langgraph.graph import StateGraph
from logitgate import LogitGate
from logitgate.integrations.langgraph import as_conditional_edge

class AgentState(TypedDict):
    content: str

# Fast binary routing edge
safety_gate = LogitGate(rubric="Is the generated response safe and appropriate to send to the user?")

# Creates a fast routing function extracting 'content' from graph state
route_verdict = as_conditional_edge(
    safety_gate,
    pass_node="publish_response",
    fail_node="human_escalation",
)

workflow = StateGraph(AgentState)
workflow.add_conditional_edges("generator", route_verdict)
```

With `ChoiceGate`, multi-way branching routes seamlessly across arbitrary node names:

```python
from logitgate import ChoiceGate
from logitgate.integrations.langgraph import as_conditional_edge

router_gate = ChoiceGate(
    question="Select the specialized agent node to handle this customer intent:",
    options={
        "A": "Billing & Refund Questions",
        "B": "Technical Troubleshooting & Bug Reports",
        "C": "Account Security & Credentials",
    },
)

route_intent = as_conditional_edge(
    router_gate,
    node_mapping={
        "A": "billing_agent",
        "B": "tech_support_agent",
        "C": "security_agent",
    },
)

workflow.add_conditional_edges("triage_agent", route_intent)
```

---

## 4. Bring-Your-Own (BYO) Syntactic & Schema Validators (`ASTGate`)

`ASTGate` (and its alias `ContractGate`) acts as a universal Layer 0 (`<0.1 ms`, `can_fast_path = False`) integration surface between your `Cascade` pipeline and any syntactic parser or schema validator in the Python ecosystem.

Any positional validator passed to `ASTGate(*validators)` follows a universal contract:
- **Pass**: Returning `True`, `None` (standard for `ast.parse`, `jsonschema.validate`), or any non-`False` parsed object (such as a Pydantic model instance, a `sqlglot` AST, or a `dict`).
- **Fail**: Returning `False`, `(False, "reason")`, or raising any `Exception` (the exception type and message are captured in `verdict.reason`).

### Handling Realistic LLM Output Wrappers (`normalize=True`)

In production, LLMs frequently wrap valid code or JSON inside `<think>...</think>` blocks, markdown code fences (` ```json ... ``` `), or conversational preamble (`"Sure! Here is the payload: {...}"`). Enable `normalize=True` (or pass a custom `normalize=my_extractor_fn`) to strip wrapper noise before running Layer 0 validators:

```python
import ast
from pydantic import BaseModel
from logitgate import ASTGate, LogitGate, JudgeGate

class SQLQueryPayload(BaseModel):
    query: str
    read_only: bool

# Accepts a Pydantic model class directly and normalizes <think> / markdown wrappers
cascade = (
    ASTGate(SQLQueryPayload, normalize=True)
    >> LogitGate("Is the SQL query strictly read-only and free of injection risks?")
    >> JudgeGate(trigger="on_ambiguous_only")
)

raw_llm_output = (
    "<think>\nLet me format the response as requested.\n</think>\n"
    "Here is the JSON payload you requested:\n"
    '{"query": "SELECT id, email FROM users WHERE active = 1", "read_only": true}\n'
    "Hope this helps!"
)

verdict = cascade.evaluate(raw_llm_output)
assert verdict.passed
```

### Drop-In Ecosystem Recipes

```python
import ast
import json
from logitgate import ASTGate

# 1. Python standard library `ast.parse` (returns ast.Module on success, raises SyntaxError on failure)
python_gate = ASTGate(ast.parse, normalize=True)

# 2. jsonschema Draft202012Validator (or jsonschema.validate)
# import jsonschema
# schema_validator = jsonschema.Draft202012Validator({"type": "object", "required": ["id"]})
# json_gate = ASTGate(schema_validator, normalize=True)

# 3. SQLGlot dialect syntax checking (raises ParseError on invalid SQL)
# import sqlglot
# sql_gate = ASTGate(lambda s: sqlglot.transpile(s, read="postgres"), normalize=True)

# 4. Tree-sitter syntax error node detection
# def check_treesitter(code: str) -> tuple[bool, str]:
#     tree = parser.parse(code.encode("utf-8"))
#     if tree.root_node.has_error:
#         return False, "Tree-sitter detected syntax error node"
#     return True, "ok"
# ts_gate = ASTGate(check_treesitter, normalize=True)
```

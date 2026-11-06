"""Strict, parseable prompt templates for the StopWise multi-step tool agent."""

from __future__ import annotations

from typing import Any, Dict, List


SYSTEM_PROMPT_BASE = """You are StopWise, a step-by-step tool-using AI agent.
At EVERY step, you MUST output your response in this exact XML-tag format:
<think>Brief reasoning about what you know and whether you are ready to answer.</think>
<action>ACTION_NAME</action>
<args>tool arguments (or leave empty if ANSWER)</args>
<ready>0.xx</ready>

Where ACTION_NAME is one of:
- GATHER: Gather new information using a tool (`search(query)` or `read(id)` for Search QA; `calc(expr)` for Maths; `run_code(python_code)` for Code). Put the tool argument inside <args>...</args>.
- CONTINUE: Execute the next planned step using the domain tool (`search(query)`, `calc(expr)`, or `run_code(python_code)`). Put the tool argument inside <args>...</args>.
- VERIFY: Double-check or verify your current candidate answer (`read(id)` or `search(query)` to cross-check a fact, `calc(expr)` to verify a calculation, or `run_code(python_code)` to test your code). Put the tool argument inside <args>...</args>.
- ANSWER: Stop immediately and output the final answer. When you choose <action>ANSWER</action>, you MUST also include <answer>YOUR_FINAL_ANSWER</answer> at the end of your turn!

And <ready>0.xx</ready> is a float in [0.00, 1.00] indicating the probability that you can answer the task correctly right now."""


DOMAIN_INSTRUCTIONS: Dict[str, str] = {
    "search_qa": """Domain: Multi-hop Search QA.
Tools available via <args>:
- For GATHER / CONTINUE: write a concise search query inside <args>query</args> (this runs `search(query, k=3)`).
- For VERIFY: write `read:PASSAGE_ID` or a verification search query inside <args>...</args>.
- For ANSWER: put ONLY the concise entity, name, date, or yes/no answer inside <answer>...</answer> (no extra explanation).""",
    "math": """Domain: Mathematical Problem Solving.
Tools available via <args>:
- For GATHER / CONTINUE / VERIFY: write an arithmetic or algebraic expression inside <args>expression</args> (this runs exact `calc(expression)`).
- For ANSWER: put ONLY the final numeric value inside <answer>NUMBER</answer> (e.g. <answer>42</answer>).""",
    "code": """Domain: Python Programming with Unit Tests.
Tools available via <args>:
- For GATHER / CONTINUE / VERIFY: write complete, runnable Python code defining the required function inside <args>python_code</args> (this runs `run_code(code)` in the sandbox with tests).
- For ANSWER: put the complete, working Python function implementation inside <answer>```python\n...\n```</answer>.""",
}


FEW_SHOT_EXAMPLES: Dict[str, str] = {
    "search_qa": """Example trajectory:
Task: Were Scott Derrickson and Ed Wood of the same nationality?
Step 1:
<think>I need to find the nationality of Scott Derrickson and Ed Wood.</think>
<action>GATHER</action>
<args>Scott Derrickson Ed Wood nationality</args>
<ready>0.20</ready>
<result>[hp_1] Scott Derrickson: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. | [hp_2] Ed Wood: Edward Davis Wood Jr. (October 10, 1924 - December 10, 1978) was an American filmmaker, actor, and author.</result>
Step 2:
<think>Both Scott Derrickson and Ed Wood are American. I have complete evidence to answer yes.</think>
<action>ANSWER</action>
<args></args>
<ready>0.95</ready>
<answer>yes</answer>""",
    "math": """Example trajectory:
Task: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Step 1:
<think>First I compute how many clips she sold in May: 48 / 2, plus April's 48.</think>
<action>CONTINUE</action>
<args>48 + 48 / 2</args>
<ready>0.40</ready>
<result>72</result>
Step 2:
<think>The calculator confirmed 48 + 24 = 72. I am ready to give the final answer.</think>
<action>ANSWER</action>
<args></args>
<ready>0.95</ready>
<answer>72</answer>""",
    "code": """Example trajectory:
Task: Write a Python function `add_nums(a, b)` that returns the sum of two numbers.
Example test: `assert add_nums(2, 3) == 5`
Step 1:
<think>I will write the `add_nums` function and test it in the sandbox.</think>
<action>CONTINUE</action>
<args>def add_nums(a, b):\n    return a + b</args>
<ready>0.50</ready>
<result>Status: SUCCESS (All tests passed)</result>
Step 2:
<think>The function passed the sandbox unit test. I am ready to answer.</think>
<action>ANSWER</action>
<args></args>
<ready>0.98</ready>
<answer>```python
def add_nums(a, b):
    return a + b
```</answer>""",
}


def build_agent_prompt(
    task: Dict[str, Any],
    history: List[Dict[str, Any]],
    few_shot: bool = False,
    include_readiness: bool = True,
) -> str:
    """Build the full prompt representing the state at step t."""
    domain = task.get("domain", "search_qa")
    sys_prompt = SYSTEM_PROMPT_BASE
    if not include_readiness:
        sys_prompt = sys_prompt.replace("<ready>0.xx</ready>\n", "")
    domain_inst = DOMAIN_INSTRUCTIONS.get(domain, DOMAIN_INSTRUCTIONS["search_qa"])

    parts: List[str] = [sys_prompt, domain_inst]
    if few_shot and domain in FEW_SHOT_EXAMPLES:
        parts.append(FEW_SHOT_EXAMPLES[domain])

    parts.append(f"Task ({domain}): {task['question']}")

    for step_rec in history:
        s_idx = step_rec["step_index"]
        think = step_rec.get("think", "")
        action = step_rec.get("action", "CONTINUE")
        args = step_rec.get("args", "")
        ready = step_rec.get("readiness", 0.5)
        result = step_rec.get("tool_result", "")

        step_str = f"Step {s_idx}:\n<think>{think}</think>\n<action>{action}</action>\n<args>{args}</args>"
        if include_readiness:
            step_str += f"\n<ready>{ready:.2f}</ready>"
        if result:
            step_str += f"\n<result>{result}</result>"
        parts.append(step_str)

    next_idx = len(history) + 1
    parts.append(f"Step {next_idx}:\n")
    return "\n\n".join(parts)


def build_forced_answer_prompt(task: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    """Build the cut-off forced-answer prompt at step t: 'Give your final answer now'."""
    domain = task.get("domain", "search_qa")
    domain_inst = DOMAIN_INSTRUCTIONS.get(domain, DOMAIN_INSTRUCTIONS["search_qa"])
    parts: List[str] = [
        SYSTEM_PROMPT_BASE,
        domain_inst,
        f"Task ({domain}): {task['question']}",
    ]
    for step_rec in history:
        s_idx = step_rec["step_index"]
        think = step_rec.get("think", "")
        action = step_rec.get("action", "CONTINUE")
        args = step_rec.get("args", "")
        ready = step_rec.get("readiness", 0.5)
        result = step_rec.get("tool_result", "")
        step_str = (
            f"Step {s_idx}:\n<think>{think}</think>\n<action>{action}</action>\n"
            f"<args>{args}</args>\n<ready>{ready:.2f}</ready>"
        )
        if result:
            step_str += f"\n<result>{result}</result>"
        parts.append(step_str)

    if domain == "search_qa":
        fmt_hint = "Put ONLY the exact short answer (entity, name, number, or yes/no) inside <answer>...</answer>."
    elif domain == "math":
        fmt_hint = "Put ONLY the final numerical answer inside <answer>...</answer>."
    else:
        fmt_hint = "Put the complete Python function code inside <answer>```python\n...\n```</answer>."

    parts.append(
        f"Give your final answer now. Do not call any more tools. {fmt_hint}\n"
        f"<think>Giving final answer now based on information gathered so far.</think>\n"
        f"<action>ANSWER</action>\n<args></args>\n<ready>1.00</ready>\n<answer>"
    )
    return "\n\n".join(parts)

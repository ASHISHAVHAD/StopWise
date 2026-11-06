"""Multi-step StopWise agent execution loop with batched LLM generation and tool execution."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.agent.action_parser import parse_agent_output, parse_forced_answer
from src.agent.prompt import build_agent_prompt, build_forced_answer_prompt
from src.tasks.code import check_code
from src.tasks.math import check_math
from src.tasks.search_qa import check_search_qa
from src.tools.calculator import calc
from src.tools.retriever import HybridRetriever
from src.tools.sandbox import run_code


class LLMGenerator:
    """Batched HuggingFace + PEFT generator with automatic left-padding and token counting."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.device = device
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model
        self.model.eval()

    def generate_batch(
        self,
        prompts: Sequence[str],
        max_new_tokens: int = 160,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_confidence: bool = False,
        batch_size: int = 24,
        num_return_sequences: int = 1,
    ) -> List[Dict[str, Any]]:
        """Generate completions for a batch of prompts and return text, token counts, and mean token prob."""
        results: List[Dict[str, Any]] = []
        eff_batch_size = max(1, batch_size // max(1, num_return_sequences))
        for start in range(0, len(prompts), eff_batch_size):
            sub_prompts = list(prompts[start : start + eff_batch_size])
            try:
                results.extend(
                    self._generate_subbatch(
                        sub_prompts,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        return_confidence=return_confidence,
                        num_return_sequences=num_return_sequences,
                    )
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                half = max(1, len(sub_prompts) // 2)
                for h_start in range(0, len(sub_prompts), half):
                    results.extend(
                        self._generate_subbatch(
                            sub_prompts[h_start : h_start + half],
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            return_confidence=return_confidence,
                            num_return_sequences=num_return_sequences,
                        )
                    )
                torch.cuda.empty_cache()
        return results

    def _generate_subbatch(
        self,
        sub_prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        return_confidence: bool,
        num_return_sequences: int = 1,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        formatted_prompts: List[str] = []
        for p in sub_prompts:
            messages = [{"role": "user", "content": p}]
            if hasattr(self.tokenizer, "apply_chat_template"):
                chat_str = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if p.rstrip().endswith("<answer>"):
                    prefix_lines = p.split("\n\n")[-1]
                    user_body = p[: -len(prefix_lines)].strip()
                    chat_str = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_body}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    chat_str += prefix_lines
                formatted_prompts.append(chat_str)
            else:
                formatted_prompts.append(p)

        enc = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)
        input_len = enc["input_ids"].shape[1]

        do_sample = temperature > 1e-4 or num_return_sequences > 1
        with torch.no_grad():
            gen_out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=return_confidence,
            )

        seqs = gen_out.sequences[:, input_len:]
        total_out = seqs.shape[0]
        for b_idx in range(total_out):
            gen_ids = seqs[b_idx]
            non_pad = (gen_ids != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            n_tokens = int(len(non_pad)) if len(non_pad) > 0 else 1
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            for stop_str in ["<result>", "\nStep "]:
                if stop_str in text:
                    text = text.split(stop_str)[0].strip()

            token_conf = 0.5
            if return_confidence and gen_out.scores:
                log_probs: List[float] = []
                for t_step, logits_step in enumerate(gen_out.scores):
                    if t_step >= len(gen_ids):
                        break
                    tok_id = int(gen_ids[t_step].item())
                    if tok_id == self.tokenizer.pad_token_id:
                        break
                    probs = torch.softmax(logits_step[b_idx].float(), dim=-1)
                    log_probs.append(float(torch.log(probs[tok_id] + 1e-12).item()))
                if log_probs:
                    token_conf = float(torch.exp(torch.tensor(sum(log_probs) / len(log_probs))).item())

            results.append(
                {
                    "text": text.strip(),
                    "tokens_used": n_tokens,
                    "token_confidence": token_conf,
                }
            )
        return results


def execute_domain_tool(
    task: Dict[str, Any],
    action: str,
    args: str,
    retriever: Optional[HybridRetriever] = None,
    top_k: int = 3,
    use_reranker: bool = True,
    sandbox_timeout: float = 5.0,
) -> str:
    """Execute the appropriate domain tool for GATHER / CONTINUE / VERIFY actions."""
    domain = task.get("domain", "search_qa")
    arg_str = str(args).strip()

    if domain == "search_qa":
        if retriever is None:
            return "Retriever not available."
        if arg_str.lower().startswith("read:") or arg_str.lower().startswith("read("):
            doc_id = arg_str.split(":", 1)[-1].strip("() ")
            r = retriever.read(doc_id)
            return f"[{r['id']}] {r['text']}"
        query = arg_str if arg_str else task["question"]
        docs = retriever.search(query=query, k=top_k, use_reranker=use_reranker)
        if not docs:
            return "No matching passages found."
        return " | ".join(f"[{d['id']}] {d['text']}" for d in docs)

    elif domain == "math":
        if not arg_str:
            return "Calculator error: empty expression in <args>."
        c = calc(arg_str)
        return c["result"] if c["ok"] else c["error"]

    elif domain == "code":
        if not arg_str:
            return "Sandbox error: empty code in <args>."
        sample_tests = task.get("test_list", [])[:1]
        return run_code(code=arg_str, tests=sample_tests, timeout_seconds=sandbox_timeout)

    return ""


def check_task_answer(task: Dict[str, Any], answer_text: str, sandbox_timeout: float = 5.0) -> Dict[str, Any]:
    """Dispatch to the domain-specific answer checker."""
    domain = task.get("domain", "search_qa")
    if domain == "search_qa":
        golds = task.get("aliases") or [task["answer"]]
        return check_search_qa(answer_text, golds)
    elif domain == "math":
        return check_math(answer_text, task["answer"])
    elif domain == "code":
        return check_code(
            prediction=answer_text,
            test_list=task.get("test_list", []),
            test_setup_code=task.get("test_setup_code", ""),
            entry_point=task.get("entry_point", ""),
            timeout_seconds=int(sandbox_timeout),
        )
    raise ValueError(f"Unknown domain: {domain}")


def run_agent_batch(
    tasks: Sequence[Dict[str, Any]],
    generator: LLMGenerator,
    retriever: Optional[HybridRetriever] = None,
    max_steps: int = 10,
    temperature: float = 0.7,
    few_shot: bool = False,
    include_readiness: bool = True,
    top_k: int = 3,
    use_reranker: bool = True,
    stop_policy: Optional[Callable[[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]], bool]] = None,
    run_id_prefix: str = "run",
    sample_idx: int = 0,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Run the StopWise multi-step agent loop across a batch of tasks in parallel steps."""
    n_tasks = len(tasks)
    histories: List[List[Dict[str, Any]]] = [[] for _ in range(n_tasks)]
    active_indices: List[int] = list(range(n_tasks))
    final_answers: List[str] = ["" for _ in range(n_tasks)]
    invalid_counts: List[int] = [0 for _ in range(n_tasks)]
    tool_calls_counts: List[int] = [0 for _ in range(n_tasks)]
    total_tokens_counts: List[int] = [0 for _ in range(n_tasks)]
    forced_at_limit: List[bool] = [False for _ in range(n_tasks)]

    for step_idx in range(1, max_steps + 1):
        if not active_indices:
            break

        step_start = time.time()
        prompts = [
            build_agent_prompt(
                tasks[i],
                histories[i],
                few_shot=few_shot,
                include_readiness=include_readiness,
            )
            for i in active_indices
        ]
        gen_outputs = generator.generate_batch(
            prompts,
            max_new_tokens=160,
            temperature=temperature,
            return_confidence=True,
            batch_size=batch_size,
        )

        # Identify any invalid actions and retry once as required by Section 3.4
        retry_local_indices: List[int] = []
        parsed_outputs: List[Dict[str, Any]] = []
        for local_idx, out in enumerate(gen_outputs):
            parsed = parse_agent_output(out["text"])
            parsed["tokens_used"] = out["tokens_used"]
            parsed["token_confidence"] = out["token_confidence"]
            parsed_outputs.append(parsed)
            if not parsed["valid_action"]:
                orig_idx = active_indices[local_idx]
                invalid_counts[orig_idx] += 1
                retry_local_indices.append(local_idx)

        if retry_local_indices:
            retry_prompts = [prompts[li] for li in retry_local_indices]
            retry_outs = generator.generate_batch(
                retry_prompts,
                max_new_tokens=160,
                temperature=max(0.3, temperature),
                return_confidence=True,
                batch_size=batch_size,
            )
            for li, r_out in zip(retry_local_indices, retry_outs):
                r_parsed = parse_agent_output(r_out["text"])
                r_parsed["tokens_used"] = parsed_outputs[li]["tokens_used"] + r_out["tokens_used"]
                r_parsed["token_confidence"] = r_out["token_confidence"]
                if not r_parsed["valid_action"]:
                    orig_idx = active_indices[li]
                    invalid_counts[orig_idx] += 1
                    # Fallback to CONTINUE with task question so trajectory can proceed
                    r_parsed["action"] = "CONTINUE"
                    r_parsed["args"] = tasks[orig_idx]["question"]
                parsed_outputs[li] = r_parsed

        next_active: List[int] = []
        step_elapsed = (time.time() - step_start) / max(1, len(active_indices))

        for local_idx, orig_idx in enumerate(active_indices):
            task = tasks[orig_idx]
            parsed = parsed_outputs[local_idx]
            state_text = prompts[local_idx]
            action = parsed["action"]
            args = parsed["args"]
            readiness = parsed["readiness"]
            tokens_used = parsed["tokens_used"]
            total_tokens_counts[orig_idx] += tokens_used

            # Check custom external stopping policy (e.g., confidence threshold or readiness rule)
            should_force_stop = False
            if stop_policy is not None and action != "ANSWER":
                should_force_stop = stop_policy(task, histories[orig_idx], parsed)

            if action == "ANSWER":
                ans = parsed["answer"]
                step_record = {
                    "task_id": task["task_id"],
                    "step_index": step_idx,
                    "state_text": state_text,
                    "think": parsed["think"],
                    "action": "ANSWER",
                    "args": args,
                    "tool_result": "",
                    "answer": ans,
                    "tokens_used": tokens_used,
                    "readiness": readiness,
                    "token_confidence": parsed["token_confidence"],
                    "valid_action": parsed["valid_action"],
                    "timing_sec": round(step_elapsed, 4),
                }
                histories[orig_idx].append(step_record)
                final_answers[orig_idx] = ans
            else:
                t0 = time.time()
                tool_res = execute_domain_tool(
                    task=task,
                    action=action,
                    args=args,
                    retriever=retriever,
                    top_k=top_k,
                    use_reranker=use_reranker,
                )
                tool_calls_counts[orig_idx] += 1
                t_tool = time.time() - t0
                step_record = {
                    "task_id": task["task_id"],
                    "step_index": step_idx,
                    "state_text": state_text,
                    "think": parsed["think"],
                    "action": action,
                    "args": args,
                    "tool_result": tool_res,
                    "answer": "",
                    "tokens_used": tokens_used,
                    "readiness": readiness,
                    "token_confidence": parsed["token_confidence"],
                    "valid_action": parsed["valid_action"],
                    "timing_sec": round(step_elapsed + t_tool, 4),
                }
                histories[orig_idx].append(step_record)
                if should_force_stop:
                    forced_at_limit[orig_idx] = True
                else:
                    next_active.append(orig_idx)

        active_indices = next_active

    # Force an answer for any task that hit max_steps or was stopped by an external policy without an answer
    need_forced_indices = [i for i in range(n_tasks) if not final_answers[i] and (i in active_indices or forced_at_limit[i])]
    if need_forced_indices:
        forced_prompts = [build_forced_answer_prompt(tasks[i], histories[i]) for i in need_forced_indices]
        forced_outs = generator.generate_batch(
            forced_prompts,
            max_new_tokens=192,
            temperature=0.2,
            return_confidence=True,
            batch_size=batch_size,
        )
        for orig_idx, f_out in zip(need_forced_indices, forced_outs):
            ans = parse_forced_answer(f_out["text"])
            final_answers[orig_idx] = ans
            total_tokens_counts[orig_idx] += f_out["tokens_used"]
            forced_at_limit[orig_idx] = True

    # Build final run records and check correctness
    run_records: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks):
        chk = check_task_answer(task, final_answers[i])
        run_records.append(
            {
                "run_id": f"{run_id_prefix}_{task['task_id']}_s{sample_idx}",
                "task_id": task["task_id"],
                "domain": task["domain"],
                "source": task.get("source", task["domain"]),
                "split": task.get("split", "test"),
                "sample_idx": sample_idx,
                "question": task["question"],
                "gold_answer": task["answer"],
                "final_answer": final_answers[i],
                "correct": int(chk["correct"]),
                "check_details": chk,
                "num_steps": len(histories[i]),
                "tool_calls": tool_calls_counts[i],
                "generated_tokens": total_tokens_counts[i],
                "invalid_actions": invalid_counts[i],
                "forced_at_limit": forced_at_limit[i],
                "steps": histories[i],
            }
        )
    return run_records


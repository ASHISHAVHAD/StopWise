"""Methods B and C: Group Relative Policy Optimization (GRPO) with `<result>` span masking."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

from src.agent.action_parser import parse_agent_output
from src.agent.loop import check_task_answer, execute_domain_tool
from src.agent.prompt import build_agent_prompt
from src.tools.retriever import HybridRetriever
from src.train.reward import compute_reward


def build_trajectory_token_mask(
    full_text: str,
    prompt_prefix: str,
    tokenizer: Any,
    max_length: int = 2048,
) -> Dict[str, torch.Tensor]:
    """Tokenize a trajectory and build `action_mask` that is 1 ONLY on agent-generated tokens
    and 0 on the initial task prompt AND on all `<result>...</result>` tool output spans.

    As specified in Section 4, Stage 6: 'do not train on tool-result text (mask `<result>` spans)'.
    """
    # Split trajectory after prompt_prefix into segments outside and inside <result>...</result>
    input_ids: List[int] = []
    action_mask: List[int] = []

    p_ids = tokenizer.encode(prompt_prefix, add_special_tokens=False)
    input_ids.extend(p_ids)
    action_mask.extend([0] * len(p_ids))

    remainder = full_text[len(prompt_prefix) :] if full_text.startswith(prompt_prefix) else full_text
    cursor = 0
    while cursor < len(remainder):
        res_start = remainder.find("<result>", cursor)
        if res_start == -1:
            agent_chunk = remainder[cursor:]
            c_ids = tokenizer.encode(agent_chunk, add_special_tokens=False)
            input_ids.extend(c_ids)
            action_mask.extend([1] * len(c_ids))
            break
        else:
            if res_start > cursor:
                agent_chunk = remainder[cursor:res_start]
                c_ids = tokenizer.encode(agent_chunk, add_special_tokens=False)
                input_ids.extend(c_ids)
                action_mask.extend([1] * len(c_ids))

            res_end = remainder.find("</result>", res_start)
            if res_end == -1:
                tool_chunk = remainder[res_start:]
                t_ids = tokenizer.encode(tool_chunk, add_special_tokens=False)
                input_ids.extend(t_ids)
                action_mask.extend([0] * len(t_ids))
                break
            else:
                res_end_full = res_end + len("</result>")
                # Also mask the following "\n\nStep k:\n" header if present
                after_res = remainder[res_end_full:]
                step_hdr_len = 0
                if after_res.startswith("\n\nStep "):
                    nl_pos = after_res.find(":\n")
                    if nl_pos != -1:
                        step_hdr_len = nl_pos + 2
                tool_chunk = remainder[res_start : res_end_full + step_hdr_len]
                t_ids = tokenizer.encode(tool_chunk, add_special_tokens=False)
                input_ids.extend(t_ids)
                action_mask.extend([0] * len(t_ids))
                cursor = res_end_full + step_hdr_len

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        action_mask = action_mask[-max_length:]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "action_mask": torch.tensor(action_mask, dtype=torch.float32),
    }


def compute_grpo_group_advantages(rewards: Sequence[float], eps: float = 1e-6) -> List[float]:
    """Compute group-normalized advantages: A_i = (R_i - mean(R)) / (std(R) + eps)."""
    if not rewards:
        return []
    t = torch.tensor(rewards, dtype=torch.float32)
    if len(rewards) == 1:
        return [0.0]
    mean = t.mean()
    std = t.std(unbiased=False)
    if float(std.item()) < eps:
        return [0.0 for _ in rewards]
    adv = (t - mean) / (std + eps)
    return [float(x) for x in adv.tolist()]


def rollout_group_trajectories(
    task: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    retriever: Optional[HybridRetriever],
    group_size: int = 8,
    max_steps: int = 6,
    temperature: float = 0.8,
    lam: float = 0.1,
    max_tokens: int = 4000,
    effort_mode: str = "full",
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Roll out `group_size` trajectories for a single task using the current policy."""
    model.eval()
    tokenizer.padding_side = "left"

    histories: List[List[Dict[str, Any]]] = [[] for _ in range(group_size)]
    active = list(range(group_size))
    final_answers = ["" for _ in range(group_size)]
    invalid_counts = [0 for _ in range(group_size)]
    tool_calls = [0 for _ in range(group_size)]
    gen_tokens = [0 for _ in range(group_size)]

    initial_prompt = build_agent_prompt(task, [])
    trajectory_texts = [initial_prompt for _ in range(group_size)]

    for step_idx in range(1, max_steps + 1):
        if not active:
            break
        prompts = [build_agent_prompt(task, histories[i]) for i in active]
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        enc = tokenizer(chat_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        in_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=160,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        next_active: List[int] = []
        for local_i, orig_i in enumerate(active):
            seq = gen[local_i, in_len:]
            non_pad = (seq != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            n_tok = int(len(non_pad)) if len(non_pad) > 0 else 1
            gen_tokens[orig_i] += n_tok
            raw_txt = tokenizer.decode(seq, skip_special_tokens=True)
            for stop_s in ["<result>", "\nStep "]:
                if stop_s in raw_txt:
                    raw_txt = raw_txt.split(stop_s)[0].strip()

            parsed = parse_agent_output(raw_txt)
            if not parsed["valid_action"]:
                invalid_counts[orig_i] += 1
                parsed["action"] = "CONTINUE"
                parsed["args"] = task["question"]

            act = parsed["action"]
            args = parsed["args"]
            r_score = parsed["readiness"]

            if act == "ANSWER":
                ans = parsed["answer"]
                final_answers[orig_i] = ans
                step_txt = (
                    f"<think>{parsed['think']}</think>\n<action>ANSWER</action>\n"
                    f"<args></args>\n<ready>{r_score:.2f}</ready>\n<answer>{ans}</answer>"
                )
                trajectory_texts[orig_i] += step_txt
                histories[orig_i].append(
                    {
                        "step_index": step_idx,
                        "think": parsed["think"],
                        "action": "ANSWER",
                        "args": "",
                        "readiness": r_score,
                        "tool_result": "",
                    }
                )
            else:
                t_res = execute_domain_tool(task, act, args, retriever=retriever, top_k=3)
                tool_calls[orig_i] += 1
                step_txt = (
                    f"<think>{parsed['think']}</think>\n<action>{act}</action>\n"
                    f"<args>{args}</args>\n<ready>{r_score:.2f}</ready>\n"
                    f"<result>{t_res}</result>\n\nStep {step_idx + 1}:\n"
                )
                trajectory_texts[orig_i] += step_txt
                histories[orig_i].append(
                    {
                        "step_index": step_idx,
                        "think": parsed["think"],
                        "action": act,
                        "args": args,
                        "readiness": r_score,
                        "tool_result": t_res,
                    }
                )
                if step_idx < max_steps:
                    next_active.append(orig_i)
        active = next_active

    rollouts: List[Dict[str, Any]] = []
    for i in range(group_size):
        chk = check_task_answer(task, final_answers[i])
        rew_info = compute_reward(
            correct=int(chk["correct"]),
            tool_calls=tool_calls[i],
            generated_tokens=gen_tokens[i],
            invalid_actions=invalid_counts[i],
            lam=lam,
            max_steps=max_steps,
            max_tokens=max_tokens,
            effort_mode=effort_mode,
        )
        n_steps = max(1, len(histories[i]))
        answered_step_1 = int(n_steps == 1 and histories[i][0]["action"] == "ANSWER")
        rollouts.append(
            {
                "prompt_prefix": initial_prompt,
                "full_text": trajectory_texts[i],
                "reward": rew_info["reward"],
                "correct": int(chk["correct"]),
                "effort": rew_info["effort"],
                "num_steps": n_steps,
                "tool_calls": tool_calls[i],
                "generated_tokens": gen_tokens[i],
                "invalid_actions": invalid_counts[i],
                "early_answer_step1": answered_step_1,
            }
        )
    return rollouts


def train_grpo_model(
    train_tasks: Sequence[Dict[str, Any]],
    val_tasks: Sequence[Dict[str, Any]],
    output_dir: str | Path,
    retriever: Optional[HybridRetriever] = None,
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    init_adapter_path: Optional[str | Path] = None,
    lam: float = 0.1,
    group_size: int = 8,
    kl_coef: float = 0.01,
    lr: float = 1e-5,
    max_steps: int = 120,
    save_every: int = 40,
    agent_max_steps: int = 6,
    effort_mode: str = "full",
    seed: int = 0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run GRPO training (Method B when init_adapter_path is None; Method C when starting from Method A)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_file = out_path / "grpo_metrics.json"
    if (out_path / "adapter_config.json").exists() and metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

    rng = random.Random(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    base_model.gradient_checkpointing_enable()

    if init_adapter_path is not None and Path(init_adapter_path).exists():
        model = PeftModel.from_pretrained(base_model, str(init_adapter_path), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_cfg)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)

    step_logs: List[Dict[str, Any]] = []
    best_running_reward = -1e9

    for step in range(1, max_steps + 1):
        task = train_tasks[(step - 1) % len(train_tasks)]
        rollouts = rollout_group_trajectories(
            task=task,
            model=model,
            tokenizer=tokenizer,
            retriever=retriever,
            group_size=group_size,
            max_steps=agent_max_steps,
            temperature=0.8,
            lam=lam,
            effort_mode=effort_mode,
            device=device,
        )

        rewards = [r["reward"] for r in rollouts]
        advantages = compute_grpo_group_advantages(rewards)

        # Perform policy gradient + KL penalty update on trajectories with non-zero advantage
        model.train()
        optimizer.zero_grad()
        total_loss_val = 0.0
        active_Rollouts = 0

        for r_item, adv in zip(rollouts, advantages):
            if abs(adv) < 1e-5:
                continue
            masked = build_trajectory_token_mask(
                full_text=r_item["full_text"],
                prompt_prefix=r_item["prompt_prefix"],
                tokenizer=tokenizer,
                max_length=1536,
            )
            in_ids = masked["input_ids"].unsqueeze(0).to(device)
            act_mask = masked["action_mask"].unsqueeze(0).to(device)
            if in_ids.size(1) < 4 or act_mask[:, 1:].sum().item() < 1.0:
                continue

            # Reference logprobs with adapter disabled (for KL penalty)
            with torch.no_grad():
                with model.disable_adapter():
                    ref_logits = model(input_ids=in_ids).logits[:, :-1, :]
                    ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
                    target_ids = in_ids[:, 1:].unsqueeze(-1)
                    ref_token_logp = ref_log_probs.gather(-1, target_ids).squeeze(-1)

            logits = model(input_ids=in_ids).logits[:, :-1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            token_logp = log_probs.gather(-1, in_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

            mask = act_mask[:, 1:]
            # Approximate per-token KL: exp(ref - cur) - (ref - cur) - 1
            diff = ref_token_logp - token_logp
            kl_per_token = torch.exp(torch.clamp(diff, -5.0, 5.0)) - diff - 1.0

            pg_loss_per_token = -float(adv) * token_logp + float(kl_coef) * kl_per_token
            traj_loss = (pg_loss_per_token * mask).sum() / (mask.sum() + 1e-6)
            (traj_loss / group_size).backward()
            total_loss_val += float(traj_loss.item())
            active_Rollouts += 1

        if active_Rollouts > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        mean_rew = sum(rewards) / len(rewards)
        mean_corr = sum(r["correct"] for r in rollouts) / len(rollouts)
        mean_steps_r = sum(r["num_steps"] for r in rollouts) / len(rollouts)
        early_rate = sum(r["early_answer_step1"] for r in rollouts) / len(rollouts)

        step_logs.append(
            {
                "step": step,
                "reward": round(mean_rew, 4),
                "correct_rate": round(mean_corr, 4),
                "mean_steps": round(mean_steps_r, 4),
                "early_answer_rate": round(early_rate, 4),
                "loss": round(total_loss_val / max(1, active_Rollouts), 4),
            }
        )

        if step % save_every == 0 or step == max_steps:
            ckpt_dir = out_path / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            recent_rew = sum(x["reward"] for x in step_logs[-min(20, len(step_logs)) :]) / min(20, len(step_logs))
            if recent_rew >= best_running_reward:
                best_running_reward = recent_rew
                model.save_pretrained(str(out_path))
                tokenizer.save_pretrained(str(out_path))

    # Ensure final best model is saved at root of out_path
    if not (out_path / "adapter_config.json").exists():
        model.save_pretrained(str(out_path))
        tokenizer.save_pretrained(str(out_path))

    tail = step_logs[-min(30, len(step_logs)) :]
    summary = {
        "lambda": lam,
        "effort_mode": effort_mode,
        "group_size": group_size,
        "kl_coef": kl_coef,
        "max_steps": max_steps,
        "init_adapter": str(init_adapter_path) if init_adapter_path else "base",
        "mean_reward_last30": round(sum(x["reward"] for x in tail) / max(1, len(tail)), 4),
        "mean_correct_last30": round(sum(x["correct_rate"] for x in tail) / max(1, len(tail)), 4),
        "mean_steps_last30": round(sum(x["mean_steps"] for x in tail) / max(1, len(tail)), 4),
        "early_answer_rate_last30": round(sum(x["early_answer_rate"] for x in tail) / max(1, len(tail)), 4),
        "history": step_logs,
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    del model
    del base_model
    torch.cuda.empty_cache()
    return summary

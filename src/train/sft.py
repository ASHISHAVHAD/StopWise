"""Method A: Supervised Fine-Tuning (SFT) on cut-off labels with 50/50 class balancing and LoRA."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from src.agent.action_parser import parse_agent_output
from src.agent.prompt import build_agent_prompt


def round_readiness(val: float, step: float = 0.1) -> float:
    """Round readiness fraction to nearest 0.1 in [0.0, 1.0]."""
    r = round(float(val) / step) * step
    return float(min(1.0, max(0.0, round(r, 2))))


def build_sft_examples_from_runs(
    labelled_runs: Sequence[Dict[str, Any]],
    task_map: Dict[str, Dict[str, Any]],
    balance_classes: bool = True,
    include_readiness: bool = True,
    max_cutoff_points: Optional[int] = None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Construct SFT training examples from successful runs and their cut-off labels.

    Specification (Section 4, Stage 5):
    - Filter for successful runs (`run['correct'] == 1`) plus high-confidence ready cut-offs.
    - Input = state at step t (after step t's tool execution, deciding step t+1, or before step t).
      Specifically: at cut-off step t (1-indexed), the prefix has steps 1..t with their tool results.
      If step t is labelled `ready` (`ready_fraction_t >= 0.6`), target action is `ANSWER` + final answer.
      Otherwise (`not_ready`), target action is the action the successful run took next (`steps[t]`).
    - Target readiness = `ready_fraction_t` rounded to 0.1.
    - Balance `ready` and `not_ready` 50/50 when `balance_classes=True`.
    """
    rng = random.Random(seed)
    ready_examples: List[Dict[str, Any]] = []
    not_ready_examples: List[Dict[str, Any]] = []

    for run in labelled_runs:
        if int(run.get("correct", 0)) != 1:
            continue
        task_id = run["task_id"]
        if task_id not in task_map:
            continue
        task = task_map[task_id]
        steps = run.get("steps", [])
        cutoffs = run.get("cutoff_labels", [])
        if not steps:
            continue

        # Also include step 0 (initial state before any tool calls): not_ready if step 1 wasn't immediate ANSWER
        if len(steps) > 1 and steps[0]["action"] != "ANSWER":
            init_prompt = build_agent_prompt(task, [], include_readiness=include_readiness)
            first_step = steps[0]
            target_r = 0.1
            target_str = (
                f"<think>{first_step.get('think', 'Gather initial information to solve the task.')}</think>\n"
                f"<action>{first_step['action']}</action>\n"
                f"<args>{first_step.get('args', '')}</args>"
            )
            if include_readiness:
                target_str += f"\n<ready>{target_r:.2f}</ready>"
            not_ready_examples.append(
                {
                    "task_id": task_id,
                    "run_id": run["run_id"],
                    "domain": run["domain"],
                    "step_index": 0,
                    "label": "not_ready",
                    "target_action": first_step["action"],
                    "target_readiness": target_r,
                    "prompt": init_prompt,
                    "completion": target_str,
                }
            )

        selected_cutoffs = list(enumerate(cutoffs))
        if max_cutoff_points is not None and len(selected_cutoffs) > max_cutoff_points:
            if max_cutoff_points == 1:
                selected_cutoffs = [selected_cutoffs[-1]]
            else:
                idxs = sorted(
                    list(
                        {
                            round(i * (len(selected_cutoffs) - 1) / (max_cutoff_points - 1))
                            for i in range(max_cutoff_points)
                        }
                    )
                )
                selected_cutoffs = [selected_cutoffs[i] for i in idxs]

        for c_idx, c_info in selected_cutoffs:
            if c_idx >= len(steps):
                continue
            prefix_history = steps[: c_idx + 1]
            state_prompt = build_agent_prompt(task, prefix_history, include_readiness=include_readiness)
            rf = float(c_info.get("ready_fraction", 0.0))
            rounded_rf = round_readiness(rf, 0.1)
            is_ready = c_info.get("label") == "ready" or rf >= 0.6

            if is_ready:
                final_ans = run.get("final_answer") or task["answer"]
                think_text = "I now have sufficient verified information to provide the final answer."
                target_str = f"<think>{think_text}</think>\n<action>ANSWER</action>\n<args></args>"
                if include_readiness:
                    target_str += f"\n<ready>{max(0.6, rounded_rf):.2f}</ready>"
                target_str += f"\n<answer>{final_ans}</answer>"
                ready_examples.append(
                    {
                        "task_id": task_id,
                        "run_id": run["run_id"],
                        "domain": run["domain"],
                        "step_index": c_info["step_index"],
                        "label": "ready",
                        "target_action": "ANSWER",
                        "target_readiness": max(0.6, rounded_rf),
                        "prompt": state_prompt,
                        "completion": target_str,
                    }
                )
            else:
                # Take the next step that the successful run actually executed
                if c_idx + 1 < len(steps):
                    next_s = steps[c_idx + 1]
                    next_act = next_s["action"]
                    if next_act == "ANSWER":
                        # If the run answered right after a not_ready step, skip or treat as CONTINUE
                        continue
                    think_text = next_s.get("think") or "I need more information before answering."
                    target_str = (
                        f"<think>{think_text}</think>\n"
                        f"<action>{next_act}</action>\n"
                        f"<args>{next_s.get('args', '')}</args>"
                    )
                    if include_readiness:
                        target_str += f"\n<ready>{min(0.5, rounded_rf):.2f}</ready>"
                    not_ready_examples.append(
                        {
                            "task_id": task_id,
                            "run_id": run["run_id"],
                            "domain": run["domain"],
                            "step_index": c_info["step_index"],
                            "label": "not_ready",
                            "target_action": next_act,
                            "target_readiness": min(0.5, rounded_rf),
                            "prompt": state_prompt,
                            "completion": target_str,
                        }
                    )

    if not balance_classes:
        all_ex = ready_examples + not_ready_examples
        rng.shuffle(all_ex)
        return all_ex

    if not ready_examples or not not_ready_examples:
        all_ex = ready_examples + not_ready_examples
        rng.shuffle(all_ex)
        return all_ex

    # Balance 50/50 by matching the larger class via oversampling the smaller class up to max(len, min*2)
    target_per_class = max(len(ready_examples), len(not_ready_examples))
    target_per_class = min(target_per_class, 2 * min(len(ready_examples), len(not_ready_examples)))

    def _sample_to_size(pool: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        if len(pool) >= k:
            return rng.sample(pool, k)
        out = list(pool)
        while len(out) < k:
            out.append(rng.choice(pool))
        return out

    balanced = _sample_to_size(ready_examples, target_per_class) + _sample_to_size(not_ready_examples, target_per_class)
    rng.shuffle(balanced)
    return balanced


class SFTDataset(Dataset):
    """Tokenized prompt-completion dataset masking prompt tokens with -100."""

    def __init__(self, examples: Sequence[Dict[str, Any]], tokenizer: Any, max_length: int = 2048) -> None:
        self.items: List[Dict[str, torch.Tensor]] = []
        for ex in examples:
            messages_prompt = [{"role": "user", "content": ex["prompt"]}]
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_str = tokenizer.apply_chat_template(
                    messages_prompt, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_str = ex["prompt"] + "\n"
            full_str = prompt_str + ex["completion"] + (tokenizer.eos_token or "")

            p_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            f_ids = tokenizer.encode(full_str, add_special_tokens=False)
            if len(f_ids) > max_length:
                # Truncate from left of prompt while keeping completion intact
                comp_len = len(f_ids) - len(p_ids)
                keep_prompt_len = max(64, max_length - comp_len)
                p_ids = p_ids[-keep_prompt_len:]
                f_ids = p_ids + f_ids[-comp_len:]

            labels = [-100] * len(p_ids) + f_ids[len(p_ids) :]
            self.items.append(
                {
                    "input_ids": torch.tensor(f_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


def _collate_sft(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["input_ids"].size(0)
        input_ids[i, :L] = item["input_ids"]
        attention_mask[i, :L] = 1
        labels[i, :L] = item["labels"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_sft_model(
    train_examples: Sequence[Dict[str, Any]],
    val_examples: Sequence[Dict[str, Any]],
    output_dir: str | Path,
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    epochs: int = 3,
    lr: float = 1e-4,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    seed: int = 0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Train Method A (LoRA SFT) on cut-off examples and evaluate held-out action accuracy & readiness error."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_file = out_path / "sft_metrics.json"
    if (out_path / "adapter_config.json").exists() and metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

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

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.train()

    train_ds = SFTDataset(train_examples, tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_Batch = max(1, len(train_ds) // batch_size)
    total_steps = max(1, (total_Batch * epochs) // grad_accum_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    step_losses: List[float] = []
    indices = list(range(len(train_ds)))
    rng = random.Random(seed)

    optimizer.zero_grad()
    global_step = 0
    for ep in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_items = [train_ds[i] for i in indices[start : start + batch_size]]
            batch = _collate_sft(batch_items, tokenizer.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            step_losses.append(float(outputs.loss.item()))

            if ((start // batch_size) + 1) % grad_accum_steps == 0 or (start + batch_size) >= len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    # Evaluate on held-out val_examples for action accuracy and readiness MAE
    model.eval()
    tokenizer.padding_side = "left"
    eval_slice = list(val_examples)[:100]
    correct_action = 0
    correct_stop_binary = 0
    readiness_errors: List[float] = []

    for start in range(0, len(eval_slice), 16):
        sub = eval_slice[start : start + 16]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in sub
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for b_i, ex in enumerate(sub):
            txt = tokenizer.decode(gen[b_i, in_len:], skip_special_tokens=True)
            parsed = parse_agent_output(txt)
            pred_act = parsed["action"]
            gold_act = ex["target_action"]
            if pred_act == gold_act:
                correct_action += 1
            if (pred_act == "ANSWER") == (gold_act == "ANSWER"):
                correct_stop_binary += 1
            readiness_errors.append(abs(float(parsed["readiness"]) - float(ex["target_readiness"])))

    n_eval = max(1, len(eval_slice))
    metrics = {
        "train_examples": len(train_examples),
        "val_examples": len(eval_slice),
        "epochs": epochs,
        "global_steps": global_step,
        "final_train_loss": round(sum(step_losses[-20:]) / max(1, len(step_losses[-20:])), 4) if step_losses else 0.0,
        "val_action_accuracy": round(correct_action / n_eval, 4),
        "val_stop_binary_accuracy": round(correct_stop_binary / n_eval, 4),
        "val_readiness_mae": round(sum(readiness_errors) / max(1, len(readiness_errors)), 4),
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    del model
    del base_model
    torch.cuda.empty_cache()
    return metrics

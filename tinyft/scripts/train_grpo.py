import argparse
import sys
import os
from pathlib import Path
import importlib
import json
from typing import Any, Callable, Dict, List, Optional

import yaml

# Add parent directory to path to import tinyft
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyft import TinyGRPOTrainer, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO training with TinyFT (YAML-First Configuration)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Basic usage with config file
  tinyft-train-grpo --config configs/grpo_example.yaml

  # Quick override for experimentation
  tinyft-train-grpo --config configs/grpo_example.yaml --output_dir ./grpo_out --run_name test
        """,
    )

    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    # Runtime overrides
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--run_name", type=str, help="Override run name for logging")
    parser.add_argument("--seed", type=int, help="Override random seed")

    # Modes
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    parser.add_argument("--dry_run", action="store_true", help="Validate configuration and exit")

    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError("Configuration file is empty or invalid")
    return cfg


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.output_dir is not None:
        cfg.setdefault("logging", {})["output_dir"] = args.output_dir
    if args.run_name is not None:
        cfg.setdefault("logging", {})["run_name"] = args.run_name
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg["debug"] = args.debug
    cfg["dry_run"] = args.dry_run
    return cfg


def print_summary(cfg: Dict[str, Any]) -> None:
    print("TinyFT GRPO Training\n")
    model_name = cfg.get("model", {}).get("model_name", "<unset>")
    tokenizer_name = cfg.get("model", {}).get("tokenizer_name", model_name)
    data = cfg.get("data", {})
    print(f"Model: {model_name}")
    print(f"Tokenizer: {tokenizer_name}")
    if "prompts_path" in data:
        print(f"Prompts Path: {data['prompts_path']}")
    else:
        print(f"Prompts: inline list ({len(data.get('prompts', []))} items)")
    log_cfg = cfg.get("logging", {})
    print(f"Output Dir: {log_cfg.get('output_dir', './outputs')}")
    print(f"Run Name: {log_cfg.get('run_name', 'auto')}")


def build_reward_fn(cfg: Dict[str, Any]) -> Callable[[str, str, Dict[str, Any]], Any]:
    rcfg = cfg.get("reward", {})
    rtype = (rcfg.get("type") or "keyword").lower()

    if rtype in {"module", "python"}:
        path = rcfg.get("path")
        if not path or ":" not in path:
            raise ValueError("reward.path must be 'module.submodule:function'")
        mod_name, func_name = path.split(":", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, func_name)
        if not callable(fn):
            raise ValueError("Loaded reward function is not callable")
        return fn

    # Built-in 'keyword' reward: optional keywords list and length bonus
    keywords: List[str] = rcfg.get("keywords", [])
    length_bonus = float(rcfg.get("length_bonus_per_word", 0.0))
    max_words = int(rcfg.get("max_words", 50))

    def reward_fn(response: str, prompt: str, ctx: Dict[str, Any]) -> Dict[str, float]:
        score = 0.0
        if keywords:
            resp_l = response.lower()
            score += sum(1.0 for kw in keywords if kw.lower() in resp_l)
        if length_bonus > 0:
            score += length_bonus * min(len(response.split()), max_words)
        return {"reward": float(score)}

    return reward_fn


def load_prompts(cfg: Dict[str, Any]) -> List[Any]:
    data = cfg.get("data", {})
    if "prompts" in data and data["prompts"]:
        return data["prompts"]
    path = data.get("prompts_path")
    if not path:
        raise ValueError("Provide either data.prompts (list) or data.prompts_path")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts path not found: {path}")
    items: List[Any] = []
    if p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # treat as raw string prompt
                    obj = {"prompt": line}
                items.append(obj)
    else:
        # treat each non-empty line as a plain prompt string
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(line)
    return items


def load_model_and_tokenizer(cfg: Dict[str, Any]):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise ImportError("transformers is required for GRPO CLI") from e

    mcfg = cfg.get("model", {})
    model_name = mcfg.get("model_name")
    if not model_name:
        raise ValueError("model.model_name is required")
    tokenizer_name = mcfg.get("tokenizer_name")
    # If tokenizer_name is None/null, use the model_name
    if tokenizer_name is None:
        tokenizer_name = model_name

    print("Loading model and tokenizer...")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    # Ensure pad token exists
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tok


def main():
    args = parse_args()

    try:
        cfg = load_config(args.config)
        cfg = apply_overrides(cfg, args)
        print_summary(cfg)

        if cfg.get("dry_run"):
            print("Configuration validation passed. Exiting (dry run).")
            return 0

        # Setup logging (TinyGRPOTrainer also sets up; do a light logger here)
        log_cfg = cfg.get("logging", {})
        backend = log_cfg.get("backend", "tensorboard")
        run_name = log_cfg.get("run_name")
        output_dir = log_cfg.get("output_dir", "./outputs/grpo")
        debug = cfg.get("debug", False)
        _ = setup_logging(
            backend=backend,
            project_name="tinyft_grpo_cli",
            run_name=run_name,
            log_dir=str(Path(output_dir) / "logs"),
            log_level="DEBUG" if debug else "INFO",
        )

        # Build reward and prompts
        reward_fn = build_reward_fn(cfg)
        prompts = load_prompts(cfg)

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(cfg)

        # Map training args
        tcfg = cfg.get("training", {})
        trainer = TinyGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            reward_fn=reward_fn,
            max_gen_len=int(tcfg.get("max_gen_len", 64)),
            num_questions_per_batch=int(tcfg.get("num_questions_per_batch", 2)),
            num_answers_per_question=int(tcfg.get("num_answers_per_question", 2)),
            temperature=float(tcfg.get("temperature", 1.0)),
            top_k=int(tcfg.get("top_k", 0)),
            top_p=float(tcfg.get("top_p", 1.0)),
            learning_rate=float(tcfg.get("learning_rate", 1e-5)),
            weight_decay=float(tcfg.get("weight_decay", 0.0)),
            adam_betas=tuple(tcfg.get("adam_betas", (0.9, 0.999))),
            adam_eps=float(tcfg.get("adam_eps", 1e-8)),
            max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
            micro_batch_size=int(tcfg.get("micro_batch_size", 8)),
            total_steps=int(tcfg.get("total_steps", 100)),
            fp16=bool(tcfg.get("fp16", False)),
            bf16=bool(tcfg.get("bf16", True)),
            logging_backend=backend,
            logging_steps=int(log_cfg.get("logging_steps", 10)),
            output_dir=str(output_dir),
            run_name=run_name,
            seed=int(cfg.get("seed", 42)),
        )

        print("Starting GRPO training...")
        trainer.train()
        print("GRPO training completed.")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


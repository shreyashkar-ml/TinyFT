"""
Qwen3-4B GRPO Math Reasoning training using TinyGRPOTrainer with the open-r1/DAPO-Math-17k-Processed dataset.
"""
from typing import Any, Dict, List
import re
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyft import TinyGRPOTrainer  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import pandas as pd
from trl import SFTTrainer, SFTConfig


# Custom reasoning and solution tags (matching the notebook)
reasoning_start = "<start_working_out>"  # Acts as <think>
reasoning_end = "<end_working_out>"     # Acts as </think>
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

# Global variables for tracking prints in reward functions
PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5


def setup_chat_template(tokenizer):
    """Set up custom chat template matching the notebook."""
    chat_template = \
        "{% if messages[0]['role'] == 'system' %}"\
            "{{ messages[0]['content'] + eos_token }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% else %}"\
            "{{ '{system_prompt}' + eos_token }}"\
            "{% set loop_messages = messages %}"\
        "{% endif %}"\
        "{% for message in loop_messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ message['content'] }}"\
            "{% elif message['role'] == 'assistant' %}"\
                "{{ message['content'] + eos_token }}"\
            "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
        "{% endif %}"

    # Replace with our specific template:
    chat_template = chat_template\
        .replace("'{system_prompt}'", f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    
    tokenizer.chat_template = chat_template
    return tokenizer


def format_sft_dataset(x):
    """Format dataset entry for SFT pre-training."""
    expected_answer = x["expected_answer"]
    problem = x["problem"]

    # Remove generated <think> and </think>
    thoughts = x["generated_solution"]
    thoughts = thoughts.replace("<think>", "").replace("</think>", "")

    # Strip newlines on left and right
    thoughts = thoughts.strip()
    # Add our custom formatting
    final_prompt = \
        reasoning_start + thoughts + reasoning_end + \
        solution_start + expected_answer + solution_end
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": final_prompt},
    ]


def extract_hash_answer(text):
    """Extract answer from solution text (for compatibility)."""
    return text


def create_regex_patterns(tokenizer):
    """Create regex patterns for matching responses."""
    # Add optional EOS token matching
    solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
        "(?:" + re.escape(tokenizer.eos_token) + ")?"

    match_format = re.compile(
        rf"{reasoning_end}.*?"\
        rf"{solution_start}(.+?){solution_end_regex}"\
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL
    )

    match_numbers = re.compile(
        solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
        flags=re.MULTILINE | re.DOTALL
    )
    
    return match_format, match_numbers


def match_format_exactly(completions, **kwargs):
    """Reward function: 3 points if format is matched exactly."""
    match_format, _ = kwargs.get('regex_patterns', (None, None))
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: 
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions, **kwargs):
    """Reward function: Partial rewards for format elements."""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # No need to reward <start_working_out> since we always prepend it!
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        scores.append(score)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    """Reward function: Check if extracted answer matches expected answer."""
    match_format, _ = kwargs.get('regex_patterns', (None, None))
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        # Correct answer gets 5 points!
        if guess == true_answer:
            score += 5.0
        # Match if spaces are seen, but less reward
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            # We also reward it if the answer is close via ratios!
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1: 
                    score += 2.0
                elif ratio >= 0.8 and ratio <= 1.2: 
                    score += 1.5
                else: 
                    score -= 2.5  # Penalize wrong answers
            except:
                score -= 4.5  # Penalize
        scores.append(score)
    return scores


def check_numbers(prompts, completions, answer, **kwargs):
    """Reward function: Extract and compare numerical answers."""
    global PRINTED_TIMES, PRINT_EVERY_STEPS
    _, match_numbers = kwargs.get('regex_patterns', (None, None))
    
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    # Print only every few steps
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            '*'*20 + f"Question:\n{question}", 
            f"\nAnswer:\n{answer[0]}", 
            f"\nResponse:\n{responses[0]}", 
            f"\nExtracted:\n{extracted_responses[0]}"
        )
    PRINTED_TIMES += 1

    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-2.5)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            # Remove commas like in 123,456
            guess = float(guess.strip().replace(",", ""))
            scores.append(3.5 if guess == true_answer else -1.5)
        except:
            scores.append(0)
            continue
    return scores


def grpo_reward_function(response: str, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main reward function that combines all reward components for TinyGRPOTrainer.
    """
    # Get regex patterns and answer from context
    regex_patterns = context.get("regex_patterns")
    answer = context.get("answer", [""])
    
    # Convert to format expected by individual reward functions
    completions = [[{"content": response}]]
    prompts = [prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]]
    
    # Calculate individual rewards
    format_exact_scores = match_format_exactly(completions, regex_patterns=regex_patterns)
    format_approx_scores = match_format_approximately(completions, regex_patterns=regex_patterns)
    answer_scores = check_answer(prompts, completions, answer, regex_patterns=regex_patterns)
    number_scores = check_numbers(prompts, completions, answer, regex_patterns=regex_patterns)
    
    # Combine all scores
    total_score = (format_exact_scores[0] + format_approx_scores[0] + 
                  answer_scores[0] + number_scores[0])
    
    return {
        "reward": float(total_score),
        "reward_info": {
            "format_exact_reward": float(format_exact_scores[0]),
            "format_approx_reward": float(format_approx_scores[0]),
            "answer_reward": float(answer_scores[0]),
            "number_reward": float(number_scores[0]),
            "total_reward": float(total_score)
        }
    }


def run_sft_pretraining(model, tokenizer):
    """Run SFT pre-training to prime the model for GRPO formatting."""
    print("Loading OpenMathReasoning-mini dataset for SFT pre-training...")
    
    # Load the pre-training dataset
    sft_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
    sft_dataset = sft_dataset.to_pandas()[
        ["expected_answer", "problem", "generated_solution"]
    ]

    # Try converting to number - if not, replace with NaN
    is_number = pd.to_numeric(pd.Series(sft_dataset["expected_answer"]), errors="coerce").notnull()
    # Select only numbers
    sft_dataset = sft_dataset.iloc[np.where(is_number)[0]]

    print(f"SFT dataset size: {len(sft_dataset)}")

    # Format the dataset
    sft_dataset["Messages"] = sft_dataset.apply(format_sft_dataset, axis=1)

    # Truncate to max_seq_length/2 for pre-training
    max_seq_length = 2048
    sft_dataset["N"] = sft_dataset["Messages"].apply(
        lambda x: len(tokenizer.apply_chat_template(x))
    )
    sft_dataset = sft_dataset.loc[sft_dataset["N"] <= max_seq_length/2].copy()

    # Convert to HF dataset
    sft_dataset["text"] = tokenizer.apply_chat_template(
        sft_dataset["Messages"].values.tolist(), tokenize=False
    )
    sft_dataset = Dataset.from_pandas(sft_dataset)

    print(f"Filtered SFT dataset size: {len(sft_dataset)}")

    # Run SFT training
    print("Starting SFT pre-training...")
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sft_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )

    sft_trainer.train()
    print("SFT pre-training completed!")
    
    # Clean up
    del sft_dataset
    torch.cuda.empty_cache()
    import gc
    gc.collect()


def main() -> None:
    print("Loading Qwen3-4B-Base model...")
    
    # Load model and tokenizer
    model_name = "unsloth/Qwen3-4B-Base"
    max_seq_length = 2048
    lora_rank = 32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set up chat template
    tokenizer = setup_chat_template(tokenizer)
    
    print("Chat template example:")
    example_output = tokenizer.apply_chat_template([
        {"role": "user", "content": "What is 1+1?"},
        {"role": "assistant", "content": f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}"},
        {"role": "user", "content": "What is 2+2?"},
    ], tokenize=False, add_generation_prompt=True)
    print(example_output)
    
    # Run SFT pre-training (commented out for quick testing, uncomment for full training)
    # run_sft_pretraining(model, tokenizer)
    
    print("Loading DAPO-Math-17k-Processed dataset...")
    # Load the main GRPO training dataset
    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    
    # Map the dataset to our format
    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["prompt"]},
        ],
        "answer": extract_hash_answer(x["solution"]),
    })
    
    print(f"Dataset size: {len(dataset)}")
    print("First example:")
    print("Prompt:", dataset[0]["prompt"])
    print("Answer:", dataset[0]["answer"])
    
    # Create regex patterns
    match_format, match_numbers = create_regex_patterns(tokenizer)
    regex_patterns = (match_format, match_numbers)
    
    # Test regex patterns
    test_response = f"Let me think!{reasoning_end}{solution_start}\n2\n{solution_end}"
    print("Regex test:", match_format.findall(test_response))
    print("Number regex test:", match_numbers.findall(f"{solution_start}  0.34  {solution_end}"))
    
    # Filter dataset based on token length (top 90% to avoid truncation)
    tokenized = dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(
            x["prompt"], add_generation_prompt=True, tokenize=True
        )},
        batched=True,
    )
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    
    maximum_length = int(np.quantile(tokenized["L"], 0.9))
    print("Max Length = ", maximum_length)
    
    # Filter only samples smaller than 90% max length
    dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
    del tokenized
    
    print(f"Filtered dataset size: {len(dataset)}")
    
    # Calculate prompt and completion lengths
    max_prompt_length = maximum_length + 1  # + 1 just in case!
    max_completion_length = max_seq_length - max_prompt_length
    
    print(f"Max prompt length: {max_prompt_length}")
    print(f"Max completion length: {max_completion_length}")
    
    # Convert dataset to format expected by TinyGRPOTrainer
    training_prompts = []
    for item in dataset:
        # Add regex patterns and answer to context for reward function
        context = {
            "regex_patterns": regex_patterns,
            "answer": [item["answer"]]
        }
        training_prompts.append({
            "prompt": item["prompt"],
            **context
        })
    
    print("Starting GRPO training...")
    trainer = TinyGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        prompts=training_prompts[:100],  # Use first 100 for testing, remove limit for full training
        reward_fn=grpo_reward_function,
        max_gen_len=max_completion_length,
        num_questions_per_batch=1,
        num_answers_per_question=4,  # num_generations from notebook
        micro_batch_size=1,
        total_steps=100,  # max_steps from notebook
        learning_rate=5e-6,  # from notebook
        fp16=False,
        bf16=True,
        logging_steps=1,
        output_dir="./outputs/qwen3_4b_grpo",
        run_name="qwen3_4b_grpo_math",
    )

    print("Training...")
    metrics = trainer.train()
    print("Final metrics:", metrics)
    
    # Run inference test
    print("\n" + "="*50)
    print("INFERENCE TEST")
    print("="*50)
    
    test_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is the sqrt of 101?"}
    ]
    
    test_text = tokenizer.apply_chat_template(
        test_prompt,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    print("Test prompt:")
    print(test_text)
    print("\nGenerating response...")
    
    # Generate response
    inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=1.0,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("Generated response:")
    print(response)
    
    print("\nTraining completed! Check outputs/qwen3_4b_grpo for logs and checkpoints.")


if __name__ == "__main__":
    main()
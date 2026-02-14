import os
import re
import json
import argparse
import time
import random
import threading
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import APIError

# ==============================================================================
# 1. Configuration – Dual judge support: Azure OpenAI (preferred) or standard OpenAI (fallback)
# ==============================================================================

# Try Azure judge via model_config.py first
_USE_AZURE_JUDGE = False
try:
    from model_config import JUDGE_CONFIG
    _azure_key = JUDGE_CONFIG.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
    _azure_endpoint = JUDGE_CONFIG.get("azure_endpoint")
    _azure_model = os.getenv("JUDGE_MODEL_NAME") or JUDGE_CONFIG.get("model_name")
    if _azure_key and _azure_endpoint and _azure_model:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=_azure_endpoint,
            api_version=JUDGE_CONFIG.get("api_version") or "2025-01-01-preview",
            api_key=_azure_key,
        )
        judge_model_name = _azure_model
        _USE_AZURE_JUDGE = True
        print("Successfully initialized Azure OpenAI client for the LLM judge.")
        print(f"Using model '{judge_model_name}' as the LLM judge.")
except (ImportError, Exception) as _init_err:
    pass

# Fallback: standard OpenAI via OPENAI_API_KEY or direct assignment
if not _USE_AZURE_JUDGE:
    DIRECT_ASSIGNED_API_KEY = "YOUR_API_KEY_GOES_HERE"
    DIRECT_ASSIGNED_MODEL_NAME = "gpt-4o"

    _oai_key = DIRECT_ASSIGNED_API_KEY
    if not _oai_key or _oai_key == "YOUR_API_KEY_GOES_HERE":
        _oai_key = os.getenv("OPENAI_API_KEY")

    _oai_model = DIRECT_ASSIGNED_MODEL_NAME
    if not _oai_model:
        _oai_model = os.getenv("JUDGE_MODEL_NAME", "gpt-4o")

    if _oai_key:
        from openai import OpenAI
        client = OpenAI(api_key=_oai_key)
        judge_model_name = _oai_model
        print("Successfully initialized Standard OpenAI client.")
        print(f"Using model '{judge_model_name}' as the LLM judge.")
    else:
        client = None
        judge_model_name = None
        print("Warning: No judge API key configured. LLM judging will be unavailable.")
        print("Set JUDGE_CONFIG in model_config.py (Azure) or OPENAI_API_KEY / DIRECT_ASSIGNED_API_KEY (OpenAI).")

# Caching and Lock for the LLM judge
judge_cache = {}
cache_lock = threading.Lock()

# ==============================================================================
# 2. Helper Functions
# ==============================================================================

def extract_answer(response_text: str) -> str | None:
    """Extracts the final answer content from the <answer> tag in the response text."""
    match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def normalize_answer(text: str) -> str:
    """Simple normalization for exact match comparison (strip and lowercase)."""
    return str(text).strip().lower()


def parse_rolling_dice_two_answer(text: str) -> tuple[str | None, int | None]:
    """
    Parse model response for rolling_dice_two: (path: red/black/same, total sum: int).
    The model gives one <answer> containing both; we extract answer1 and answer2.
    """
    if not text:
        return None, None
    m = re.search(r"\b(red|black|same)\b", text, re.I)
    ans1 = m.group(1).lower() if m else None
    nums = re.findall(r"\d+", text)
    ans2 = int(nums[-1]) if nums else None
    return ans1, ans2

def judge_with_llm(question: str, ground_truth: str, model_response: str) -> bool:
    """Uses the configured LLM model to semantically judge the model's answer."""
    cache_key = (question, ground_truth, model_response)
    with cache_lock:
        if cache_key in judge_cache:
            return judge_cache[cache_key]

    judge_prompt = f"""
    You are a strict and precise evaluator. Your task is to determine if the model's final answer is correct based on the ground truth.

    Your evaluation MUST focus EXCLUSIVELY on the final answer provided within the `<answer>` and `</answer>` tags.
    IGNORE all reasoning, explanations, or any other text outside of these tags. The correctness of the reasoning process is not part of your evaluation.

    Here is the data:
    - Question: "{question}"
    - Ground Truth Answer: "{ground_truth}"
    - Model's Full Response:
    ---
    {model_response}
    ---

    Based on the ground truth, is the answer inside the `<answer>` tag correct?
    Please respond with only one word: 'Correct' or 'Incorrect'.
    """
    messages = [{"role": "user", "content": judge_prompt}]

    retries = 3
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=judge_model_name,
                messages=messages,
                max_tokens=25,
                temperature=0,
            )
            judge_result = response.choices[0].message.content.strip().lower()
            is_correct = (judge_result == 'correct')
            with cache_lock:
                judge_cache[cache_key] = is_correct
            return is_correct
        except APIError as e:
            # Catch critical API errors
            print(f"LLM API Error on retry {i+1}: {e}")
            break
        except Exception as e:
            print(f"LLM Call Failed on retry {i+1}: {e}")
            if i < retries - 1:
                time.sleep(random.uniform(1, 3))
            else:
                print("LLM judging failed after all retries.")
    return False

def process_item(item: dict, use_llm_judge: bool) -> tuple | None:
    """
    Processes a single data item.
    Returns: ( (model, task, scenario), is_correct, task_name, uid )
    """
    try:
        model_alias = item['model_alias']
        task_name = item['task_name']
        data = item['data']

        scenario = data.get("scenario", "unknown_scenario")
        model_response = data.get("response", "")
        original_data = data.get("original_data", {})
        key = (model_alias, task_name, scenario)

        # rolling_dice_two: must match BOTH answer1 and answer2 to score
        if "rolling_dice_two" in task_name:
            gt1 = original_data.get("answer1")
            gt2 = original_data.get("answer2")
            if gt1 is None or gt2 is None or not model_response:
                return None
            predicted = extract_answer(model_response)
            if predicted is None:
                return (key, False, task_name, original_data.get("uid"))
            p1, p2 = parse_rolling_dice_two_answer(predicted)
            try:
                gt2_int = int(gt2)
            except (TypeError, ValueError):
                gt2_int = None
            match1 = p1 is not None and normalize_answer(p1) == normalize_answer(str(gt1))
            match2 = p2 is not None and gt2_int is not None and p2 == gt2_int
            is_correct = match1 and match2
            return (key, is_correct, task_name, original_data.get("uid"))

        # Standard: single answer
        ground_truth = original_data.get("answer")
        question = original_data.get("question", "")
        uid = original_data.get("uid")
        if ground_truth is None or not model_response:
            return None

        predicted_answer = extract_answer(model_response)
        is_correct = False
        if predicted_answer is not None:
            is_correct = normalize_answer(predicted_answer) == normalize_answer(ground_truth)
            if not is_correct and use_llm_judge:
                is_correct = judge_with_llm(question, str(ground_truth), model_response)

        return (key, is_correct, task_name, uid)
    except Exception:
        # Silently skip items that fail processing
        return None

# ==============================================================================
# 3. Aspect grouping for reporting (task folder name -> aspect and display name)
# ==============================================================================

ASPECT_ORDER = ("Causal", "Geometry", "Puzzle", "Physics")

TASK_TO_ASPECT = {
    "paper_airplane": "Causal",
    "gear_rotation": "Causal",
    "rolling_dice_top": "Causal",
    "rolling_dice_sum": "Causal",
    "rolling_dice_two": "Causal",
    "convex_hull": "Geometry",
    "overlap": "Geometry",
    "localizer": "Geometry",
    "mirror_pattern": "Geometry",
    "cubes_count": "Geometry",
    "cubes_missing": "Geometry",
    "puzzle": "Puzzle",
    "multi_piece_puzzle": "Puzzle",
    "defuse_a_bomb": "Puzzle",
    "unfolded_cube": "Puzzle",
    "trailer_cubes_count": "Puzzle",
    "trailer_cubes_missing": "Puzzle",
    "billiards": "Physics",
    "electric_charge": "Physics",
    "mirror_clock": "Physics",
}

TASK_DISPLAY_NAMES = {
    "paper_airplane": "Paper Airplane",
    "gear_rotation": "Gear Rotation",
    "rolling_dice_top": "Rolling Dice (Top)",
    "rolling_dice_sum": "Rolling Dice (Sum)",
    "rolling_dice_two": "Rolling Dice (Two)",
    "convex_hull": "Convex Hull",
    "overlap": "Overlap",
    "localizer": "Localizer",
    "mirror_pattern": "Mirror Pattern",
    "cubes_count": "Cubes Count",
    "cubes_missing": "Cubes Missing",
    "puzzle": "Puzzle",
    "multi_piece_puzzle": "Multi-piece Puzzle",
    "defuse_a_bomb": "Defuse a Bomb",
    "unfolded_cube": "Unfolded Cube",
    "trailer_cubes_count": "Trailer Cubes Count",
    "trailer_cubes_missing": "Trailer Cubes Missing",
    "billiards": "Billiards",
    "electric_charge": "Electric Charge",
    "mirror_clock": "Mirror Clock",
}


# ==============================================================================
# 4. Main Logic
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy from benchmark result files with multi-threading.")
    parser.add_argument("-r", "--results-dir", type=str, required=True, help="Directory containing the .jsonl result files.")
    parser.add_argument("--use-llm-judge", action="store_true", help="Use LLM to judge answers that fail exact match.")
    parser.add_argument("-w", "--workers", type=int, default=10, help="Number of concurrent threads for judging.")
    args = parser.parse_args()

    # Special task names: paired scoring (both sub-answers required for one score)
    GEAR_TASK_NAME = "gear_rotation"  # 40 entries (1.1–20.2) → 20 scores; 1.1 and 1.2 both correct = 1

    results_path = Path(args.results_dir)
    if not results_path.is_dir():
        print(f"Error: Directory not found at '{results_path}'")
        return

    all_items = []
    jsonl_files = list(results_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"Warning: No .jsonl files found in '{results_path}'.")
        return

    print(f"Found {len(jsonl_files)} result files. Loading tasks...")
    for file_path in jsonl_files:
        stem = file_path.stem
        # Parse {model_alias}_{task_name}.jsonl; model_alias can contain underscores (e.g. qwen3_vl)
        model_alias, task_name = None, None
        for task in TASK_TO_ASPECT:
            suffix = "_" + task
            if stem.endswith(suffix):
                model_alias = stem[: -len(suffix)]
                task_name = task
                break
        if model_alias is None:
            parts = stem.split("_", 1)
            model_alias = parts[0]
            task_name = parts[1] if len(parts) > 1 and parts[1] else "unknown_task"
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    all_items.append({
                        'model_alias': model_alias,
                        'task_name': task_name,
                        'data': data
                    })
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode a line in {file_path}")

    if not all_items:
        print("No valid test cases found to process.")
        return

    print(f"Loaded {len(all_items)} total test cases. Starting processing with {args.workers} workers...")

    # Dictionary to store intermediate results before final scoring
    intermediate_results = defaultdict(list)

    # Check if LLM judging should be performed and if client is initialized
    llm_judging_enabled = args.use_llm_judge and client is not None

    if args.use_llm_judge and not llm_judging_enabled:
        print("LLM Judging was requested but the client failed to initialize. Proceeding with Exact Match only.")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all items for parallel processing
        future_to_item = {
            executor.submit(process_item, item, llm_judging_enabled): item
            for item in all_items
        }

        for future in tqdm(as_completed(future_to_item), total=len(all_items), desc="Processing items"):
            result = future.result()
            if result:
                key, is_correct, task_name, uid = result
                intermediate_results[key].append({
                    "correct": is_correct,
                    "task": task_name,
                    "uid": uid
                })

    # Final scoring and aggregation step
    print("\nAggregating results with special scoring logic...")
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for key, results_list in intermediate_results.items():
        model, task, scenario = key

        # gear_rotation: 40 entries (n.1, n.2 for n=1..20) → 20 scores max; both n.1 and n.2 must be correct to count question n as correct
        if GEAR_TASK_NAME in task:
            paired_results = defaultdict(dict)
            for res in results_list:
                if res['uid'] is None:
                    continue
                try:
                    uid_str = str(res['uid'])
                    if '.' not in uid_str:
                        paired_results[uid_str]['single'] = res['correct']
                        continue
                    base_uid, sub_uid = uid_str.split('.', 1)
                    paired_results[base_uid][sub_uid] = res['correct']
                except (ValueError, AttributeError):
                    print(f"Warning: Invalid UID format '{res['uid']}' in task '{task}'. Skipping.")
            total_pairs = len(paired_results)
            correct_pairs = 0
            for sub_results in paired_results.values():
                if sub_results.get('1', False) and sub_results.get('2', False):
                    correct_pairs += 1
                elif 'single' in sub_results and sub_results['single']:
                    correct_pairs += 1
            stats[key]['total'] = total_pairs
            stats[key]['correct'] = correct_pairs

        # Standard logic for all other tasks: count individual correctness
        else:
            stats[key]['total'] = len(results_list)
            stats[key]['correct'] = sum(1 for res in results_list if res['correct'])

    # ==============================================================================
    # 5. Print Results (aspect-level + task-level grouped by aspect)
    # ==============================================================================
    print("\n\n--- Benchmark Accuracy Results ---")
    if not stats:
        print("No results to display.")
        return

    # Aggregate per (model, aspect) for aspect-level reporting
    aspect_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for key, v in stats.items():
        model, task, scenario = key
        asp = TASK_TO_ASPECT.get(task, "Other")
        aspect_stats[model][asp]["correct"] += v["correct"]
        aspect_stats[model][asp]["total"] += v["total"]

    models = sorted({m for m, _t, _s in stats.keys()})
    for model in models:
        print(f"\n  --- Model: {model} ---")
        # Aspect-level accuracies
        print("  Aspect-level:")
        for asp in ASPECT_ORDER:
            t = aspect_stats[model][asp]["total"]
            if t == 0:
                continue
            c = aspect_stats[model][asp]["correct"]
            acc = (c / t * 100)
            print(f"    {asp:<10} {acc:>6.2f}%  ({c}/{t})")
        if aspect_stats[model]["Other"]["total"] > 0:
            c = aspect_stats[model]["Other"]["correct"]
            t = aspect_stats[model]["Other"]["total"]
            acc = (c / t * 100) if t > 0 else 0.0
            print(f"    {'Other':<10} {acc:>6.2f}%  ({c}/{t})")

        # Task-level (grouped by aspect)
        print("  Task-level:")
        print(f"    {'Task':<28} | {'Scenario':<15} | {'Accuracy':<10} | {'Correct/Total'}")
        print("    " + "-" * 72)
        model_keys = [k for k in stats.keys() if k[0] == model]
        for asp in ASPECT_ORDER:
            keys_in_aspect = [k for k in model_keys if TASK_TO_ASPECT.get(k[1], "Other") == asp]
            for key in sorted(keys_in_aspect, key=lambda k: (k[1], k[2])):
                _m, task, scenario = key
                correct = stats[key]["correct"]
                total = stats[key]["total"]
                acc = (correct / total * 100) if total > 0 else 0.0
                name = TASK_DISPLAY_NAMES.get(task, task)
                print(f"    {name:<28} | {scenario:<15} | {acc:>8.2f}% | ({correct}/{total})")
        # Other tasks if any
        other_keys = [k for k in model_keys if TASK_TO_ASPECT.get(k[1], "Other") == "Other"]
        for key in sorted(other_keys, key=lambda k: (k[1], k[2])):
            _m, task, scenario = key
            correct = stats[key]["correct"]
            total = stats[key]["total"]
            acc = (correct / total * 100) if total > 0 else 0.0
            print(f"    {task:<28} | {scenario:<15} | {acc:>8.2f}% | ({correct}/{total})")

if __name__ == "__main__":
    main()

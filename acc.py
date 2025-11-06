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
from openai import OpenAI, APIError 

# ==============================================================================
# 1. Configuration (Modified to allow direct assignment for testing/demand)
# ==============================================================================

DIRECT_ASSIGNED_API_KEY = "YOUR_API_KEY_GOES_HERE"
DIRECT_ASSIGNED_MODEL_NAME = "gpt-4o"

API_KEY = DIRECT_ASSIGNED_API_KEY
if not API_KEY or API_KEY == "YOUR_API_KEY_GOES_HERE":
    API_KEY = os.getenv("OPENAI_API_KEY") 

JUDGE_MODEL_NAME = DIRECT_ASSIGNED_MODEL_NAME
if not JUDGE_MODEL_NAME:
    JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "gpt-4o")

# Initialize OpenAI client
try:
    if not API_KEY:
        raise ValueError("Missing essential API key. Please set DIRECT_ASSIGNED_API_KEY or OPENAI_API_KEY.")
        
    # Standard OpenAI client initialization
    client = OpenAI(
        api_key=API_KEY,
    )
    judge_model_name = JUDGE_MODEL_NAME
    print(f"Successfully initialized Standard OpenAI client.")
    print(f"Using model '{judge_model_name}' as the LLM judge.")
except ValueError as e:
    print(f"Configuration Error: {e}")
    print("Please set the API key directly in the script or via OPENAI_API_KEY environment variable.")
    exit(1)
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")
    exit(1)

# Caching and Lock for the LLM judge
judge_cache = {}
cache_lock = threading.Lock()

# ==============================================================================
# 2. Helper Functions (Minor change in judge_with_llm to use standard model name)
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
            # Use the model name directly as required by the standard OpenAI client
            response = client.chat.completions.create(
                model=judge_model_name,
                messages=messages,
                max_tokens=10,
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
        ground_truth = original_data.get("answer")
        question = original_data.get("question", "")
        # Extract UID for special pairing logic (e.g., gear task)
        uid = original_data.get("uid")
        if ground_truth is None or not model_response:
            return None
        
        key = (model_alias, task_name, scenario)
        predicted_answer = extract_answer(model_response)
        
        is_correct = False # Default to False
        if predicted_answer is not None:
            # Check for exact match (EM) first
            is_correct = normalize_answer(predicted_answer) == normalize_answer(ground_truth)
            # If EM fails, use LLM judge if enabled
            if not is_correct and use_llm_judge:
                is_correct = judge_with_llm(question, str(ground_truth), model_response)

        # Return comprehensive info for the aggregation step
        return (key, is_correct, task_name, uid)
    except Exception:
        # Silently skip items that fail processing
        return None

# ==============================================================================
# 3. Main Logic
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy from benchmark result files with multi-threading.")
    parser.add_argument("-r", "--results-dir", type=str, required=True, help="Directory containing the .jsonl result files.")
    parser.add_argument("--use-llm-judge", action="store_true", help="Use LLM to judge answers that fail exact match.")
    parser.add_argument("-w", "--workers", type=int, default=10, help="Number of concurrent threads for judging.")
    args = parser.parse_args()

    # Define the special task name for paired scoring logic.
    GEAR_TASK_NAME = "gear_rotation" # Keeping original name for filename matching

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
        parts = file_path.stem.split('_', 1)
        model_alias = parts[0]
        # Handle case where filename might be just 'model_alias.jsonl'
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
    llm_judging_enabled = args.use_llm_judge and 'client' in globals()
    
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

        # Special pairing logic for the gear task
        if GEAR_TASK_NAME in task:
            paired_results = defaultdict(dict)
            for res in results_list:
                if res['uid'] is None: continue
                try:
                    # UID is expected to be float-like (e.g., 1.1), split into base and sub UID
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
                # A pair (1.1 and 1.2) is correct only if BOTH sub-results are True
                if sub_results.get('1', False) and sub_results.get('2', False):
                    correct_pairs += 1
                elif 'single' in sub_results and sub_results['single']:
                    correct_pairs += 1 
            
            # The 'total' here represents the number of PAIRS
            stats[key]['total'] = total_pairs
            stats[key]['correct'] = correct_pairs
        
        # Standard logic for all other tasks: count individual correctness
        else:
            stats[key]['total'] = len(results_list)
            stats[key]['correct'] = sum(1 for res in results_list if res['correct'])

    # ==============================================================================
    # 4. Print Results
    # ==============================================================================
    print("\n\n--- Benchmark Accuracy Results ---")
    if not stats:
        print("No results to display.")
        return

    sorted_keys = sorted(stats.keys())
    print(f"{'Model':<20} | {'Task':<30} | {'Scenario':<15} | {'Accuracy':<10} | {'Correct/Total'}")
    print("-" * 90)

    for key in sorted_keys:
        model, task, scenario = key
        correct = stats[key]['correct']
        total = stats[key]['total']
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Print the aggregated results
        print(f"{model:<20} | {task:<30} | {scenario:<15} | {accuracy: >8.2f}% | ({correct}/{total})")

if __name__ == "__main__":
    main()
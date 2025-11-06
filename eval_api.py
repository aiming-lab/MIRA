import os
import sys
import json
import random
import time
import base64
import argparse
import threading
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm
from PIL import Image

# ==============================================================================
# 1. Model Configuration
# ==============================================================================

MODEL_CONFIG = {
    "gpt-4o": {
        "model_name": "gpt-4o",
        "api_key": "your-api-key-here",
        "base_url": None,  # Use default OpenAI endpoint, or specify custom endpoint
        "organization": None,  # Optional: your organization ID
    },
    "gpt-4o-mini": {
        "model_name": "gpt-4o-mini",
        "api_key": "your-api-key-here",
        "base_url": None,
        "organization": None,
    },
    "gpt-4-turbo": {
        "model_name": "gpt-4-turbo",
        "api_key": "your-api-key-here",
        "base_url": None,
        "organization": None,
    },
    # Add more models as needed
    # For custom endpoints (e.g., compatible APIs):
    # "custom-model": {
    #     "model_name": "model-name",
    #     "api_key": "your-api-key",
    #     "base_url": "https://api.custom-provider.com/v1",
    #     "organization": None,
    # },
}

# ==============================================================================
# 2. Core Utility Functions
# ==============================================================================

def get_model_client(model_alias):
    """
    Initialize OpenAI client for a specific model.
    
    Args:
        model_alias: Key from MODEL_CONFIG dictionary
        
    Returns:
        Tuple of (client, model_name)
    """
    config = MODEL_CONFIG.get(model_alias)
    if not config:
        raise ValueError(f"Model alias '{model_alias}' not found in MODEL_CONFIG.")
    
    # Build client kwargs
    client_kwargs = {
        "api_key": config['api_key'],
    }
    
    if config.get('base_url'):
        client_kwargs['base_url'] = config['base_url']
    
    if config.get('organization'):
        client_kwargs['organization'] = config['organization']
    
    client = OpenAI(**client_kwargs)
    
    return client, config['model_name']


def find_image_path(directory: Path, base_filename: str) -> Path | None:
    """
    Find image file in directory with various extensions.
    
    Args:
        directory: Directory to search in
        base_filename: Base name of the image file
        
    Returns:
        Path to image file or None if not found
    """
    if not base_filename:
        return None
    
    base_name = Path(base_filename).stem
    for ext in ['.png', '.jpg', '.jpeg']:
        path = directory / (base_name + ext)
        if path.exists():
            return path
    return None


def find_cot_images(cot_dir: Path, main_image_filename: str) -> list[Path]:
    """
    Find chain-of-thought (CoT) images related to the main image.
    
    Args:
        cot_dir: Directory containing CoT images
        main_image_filename: Filename of the main question image
        
    Returns:
        List of paths to CoT images
    """
    if not main_image_filename:
        return []
    
    main_image_stem = Path(main_image_filename).stem
    pattern1 = f"{main_image_stem}.*"
    pattern2 = f"{main_image_stem}_*"
    
    found_files = list(cot_dir.glob(pattern1)) + list(cot_dir.glob(pattern2))
    image_extensions = {'.png', '.jpg', '.jpeg'}
    unique_images = {p for p in found_files if p.suffix.lower() in image_extensions}
    
    return sorted(list(unique_images))


def pil_image_to_data_url(img_path: Path) -> str | None:
    """
    Convert PIL image to base64 data URL.
    
    Args:
        img_path: Path to image file
        
    Returns:
        Base64-encoded data URL or None if error
    """
    try:
        with Image.open(img_path) as img:
            buf = BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"Error processing image {img_path}: {e}", file=sys.stderr)
        return None


def call_openai_api_with_retry(client, model_name, messages, max_retries=20):
    """
    Call OpenAI API with exponential backoff retry logic.
    
    Args:
        client: OpenAI client instance
        model_name: Name of the model to use
        messages: List of message dictionaries
        max_retries: Maximum number of retry attempts
        
    Returns:
        API response object or None if all retries failed
    """
    retries = 0
    current_delay = 1
    
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=16384,
            )
            return response
        except Exception as e:
            retries += 1
            print(
                f"\nAPI call error for {model_name}: {e}. "
                f"Retrying ({retries}/{max_retries}) in {current_delay:.2f}s...",
                file=sys.stderr
            )
            time.sleep(current_delay)
            current_delay = min(current_delay * 2 + random.uniform(0, 1), 60)
    
    return None

# ==============================================================================
# 3. Task Processing Logic
# ==============================================================================

# Prompt templates for different reasoning scenarios
PROMPT_TEMPLATES = {
    "direct_answer": (
        "Question: {question}\n\n"
        "Please provide the final answer directly. "
        "The final answer is placed in <answer></answer>."
    ),
    "text_cot": (
        "Question: {question}\n\n"
        "Please first conduct step-by-step reasoning, and then provide the final answer. "
        "The final answer is placed in <answer></answer>."
    ),
    "visual_cot": (
        "Based on the question image and the intermediate reasoning image(s) provided, "
        "please continue the reasoning to solve the problem.\n\n"
        "Question: {question}\n\n"
        "The final answer is placed in <answer></answer>."
    )
}

# Thread-safe file writing
file_locks = {}


def process_single_task(client, model_name, model_alias, data_item, task_dir, scenario, output_dir):
    """
    Process a single benchmark task.
    
    Args:
        client: OpenAI client instance
        model_name: Name of the model
        model_alias: Alias for the model (used in output filename)
        data_item: Dictionary containing task data
        task_dir: Directory containing task images
        scenario: Type of reasoning scenario (direct_answer, text_cot, visual_cot)
        output_dir: Directory to save results
    """
    question = data_item.get('question', '')
    relative_image_path = data_item.get('image_path')
    image_filename = Path(relative_image_path).name if relative_image_path else None
    
    # Build prompt
    prompt_text = PROMPT_TEMPLATES[scenario].format(question=question)
    content_list = []
    
    # Add main question image
    main_img_path = find_image_path(task_dir / "image", image_filename)
    if main_img_path:
        main_img_url = pil_image_to_data_url(main_img_path)
        if main_img_url:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": main_img_url}
            })
    elif image_filename:
        print(
            f"\nWarning: Main image '{image_filename}' not found in {task_dir / 'image'}",
            file=sys.stderr
        )
    
    # Add CoT images if visual_cot scenario
    if scenario == "visual_cot":
        cot_image_paths = find_cot_images(task_dir / "cot", image_filename)
        if not cot_image_paths and image_filename:
            print(
                f"\nWarning: No CoT image found for '{image_filename}' in {task_dir / 'cot'}",
                file=sys.stderr
            )
        
        for cot_path in cot_image_paths:
            cot_img_url = pil_image_to_data_url(cot_path)
            if cot_img_url:
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": cot_img_url}
                })
    
    # Add text prompt
    content_list.append({"type": "text", "text": prompt_text})
    
    # Call API
    messages = [{"role": "user", "content": content_list}]
    response = call_openai_api_with_retry(client, model_name, messages)
    
    # Prepare result
    result = {
        "uid": data_item.get('uid'),
        "scenario": scenario,
        "model_alias": model_alias,
        "response": response.choices[0].message.content.strip() if response else "API_CALL_FAILED",
        "original_data": data_item,
    }
    
    # Thread-safe file writing
    output_filename = f"{model_alias}_{task_dir.name}.jsonl"
    output_path = output_dir / output_filename
    lock = file_locks.setdefault(output_path, threading.Lock())
    
    with lock:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

# ==============================================================================
# 4. Model-Level Benchmark Runner
# ==============================================================================

def run_benchmark_for_model(model_alias, base_tasks, output_dir, workers_per_model, pbar_position):
    """
    Run all benchmark tasks for a single model with concurrent workers.
    
    Args:
        model_alias: Model identifier
        base_tasks: List of (data_item, task_dir, scenario) tuples
        output_dir: Directory to save results
        workers_per_model: Number of concurrent workers for this model
        pbar_position: Position for progress bar in terminal
    """
    try:
        client, model_name = get_model_client(model_alias)
    except Exception as e:
        print(f"Failed to initialize client for {model_alias}: {e}", file=sys.stderr)
        return
    
    with ThreadPoolExecutor(max_workers=workers_per_model) as executor:
        progress_bar = tqdm(
            total=len(base_tasks),
            desc=f"Model: {model_alias}",
            position=pbar_position
        )
        
        futures = {
            executor.submit(
                process_single_task,
                client, model_name, model_alias,
                data_item, task_dir, scenario,
                output_dir
            )
            for data_item, task_dir, scenario in base_tasks
        }
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(
                    f'\nTask for model {model_alias} generated an exception: {exc}',
                    file=sys.stderr
                )
            progress_bar.update(1)
        
        progress_bar.close()

# ==============================================================================
# 5. Main Entry Point
# ==============================================================================

def main():
    """
    Main function to orchestrate benchmark execution across multiple models.
    Each model runs with its own thread pool for concurrent task processing.
    """
    parser = argparse.ArgumentParser(
        description="Run benchmark with a dedicated thread pool per model."
    )
    parser.add_argument(
        "-b", "--benchmark-dir",
        type=str, required=True,
        help="Root directory of the benchmark."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str, required=True,
        help="Directory to save the result .jsonl files."
    )
    parser.add_argument(
        "-w", "--workers-per-model",
        type=int, default=1,
        help="Number of concurrent threads FOR EACH model."
    )
    args = parser.parse_args()
    
    benchmark_path = Path(args.benchmark_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load all benchmark tasks
    base_tasks = []
    jsonl_files = list(benchmark_path.rglob('*.jsonl'))
    
    if not jsonl_files:
        print(f"Error: No .jsonl files found in '{benchmark_path}'. Please check the path.")
        return
    
    print("Loading benchmark data...")
    for jsonl_file in jsonl_files:
        task_dir = jsonl_file.parent
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data_item = json.loads(line)
                    # Create tasks for each scenario
                    for scenario in PROMPT_TEMPLATES.keys():
                        base_tasks.append((data_item, task_dir, scenario))
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode a line in {jsonl_file}", file=sys.stderr)
    
    print(f"Found {len(base_tasks)} base tasks to run for each model.")
    
    model_aliases = list(MODEL_CONFIG.keys())
    num_models = len(model_aliases)
    
    # Use a main thread pool to manage parallel execution across models
    with ThreadPoolExecutor(max_workers=num_models) as main_executor:
        print(
            f"\nStarting benchmark run for {num_models} models, "
            f"each with {args.workers_per_model} worker(s)..."
        )
        
        # Submit a benchmark runner for each model
        futures = [
            main_executor.submit(
                run_benchmark_for_model,
                alias, base_tasks, output_path,
                args.workers_per_model, idx
            )
            for idx, alias in enumerate(model_aliases)
        ]
        
        # Wait for all models to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(
                    f'\nA model-level runner generated an exception: {exc}',
                    file=sys.stderr
                )
    
    print("\n\nAll benchmark runs completed.")
    print(f"Results saved in: {output_path.resolve()}")


if __name__ == "__main__":
    main()
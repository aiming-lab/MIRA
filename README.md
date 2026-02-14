<p align="center">
  <img src="assets/mira.png" width="15%" alt="MIRA teaser"> <br>
</p>

# When Visualizing is the First Step to Reasoning: MIRA, a Benchmark for Visual Chain-of-Thought.

🌟 This is the official evaluation repository for **MIRA (Multimodal Imagination for Reasoning Assessment)**, accompanying the paper: **"When Visualizing is the First Step to Reasoning: MIRA, a Benchmark for Visual Chain-of-Thought."**

[[📖 ArXiv Paper]](https://arxiv.org/abs/2511.02779)
[[🌐 Homepage]](https://mira-benchmark.github.io/)
[[🤗 Dataset]](https://huggingface.co/datasets/YiyangAiLab/MIRA)

MIRA evaluates whether MLLMs can think while drawing—i.e., generate and use intermediate **visual** representations (sketches, diagrams, trajectories) as part of reasoning.
This repo currently includes **answer extraction & evaluation** and **accuracy calculation** utilities.


## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/aiming-lab/MIRA.git
cd MIRA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Edit model_config.py — fill in your Azure OpenAI credentials
#    MODEL_CONFIG  → the model(s) you want to evaluate
#    JUDGE_CONFIG  → the judge model for LLM-based scoring
vi model_config.py

# 4. One-click: download data + evaluate + compute accuracy
./run_eval.sh -m gpt4o        # single model
# or
./run_eval.sh                  # all models in MODEL_CONFIG
```

Results will be saved to `results.txt`. That's it!

> **Note:** If you don't use Azure, you can also evaluate with standard OpenAI — see [Responses Generation](#responses-generation) below.

---

## 👀 About MIRA

- **Goal:** Test visual chain-of-thought (Visual-CoT) reasoning, not just text-only CoT.
- **Content:** 546 carefully curated items spanning 20 task types across four domains (e.g., geometry, physics-like reasoning, spatial/logical puzzles, causal transformations).
- **Eval Modes:** Direct (image + question), Text-CoT, Visual-CoT (with gold CoT images).

<p align="center">
  <img src="assets/fig1.jpg" width="90%" alt="MIRA teaser"> <br>
</p>


## 📦 Dataset Usage

### Data Downloading

You can download the dataset by the following command (Taking downloading billiards data as an example):

```python
from datasets import load_dataset
dataset = load_dataset("YiyangAiLab/MIRA", "billiards")
```

```python
from datasets import load_dataset
dataset = load_dataset("YiyangAiLab/MIRA")
```

Alternatively, use the provided download script to fetch all 20 tasks (rate-limit aware, 60 s delay between tasks):

```bash
pip install -r requirements.txt
python download_data.py
```

### Data Format

The dataset is provided in JSON Lines (jsonl) format. Each line is a standalone JSON object with the following fields:

```
{
  "uid (int)": Unique identifier for the sample,
  "image_path (string)": Relative or absolute path to the input image file,
  "question (string)": The natural-language prompt associated with the image,
  "answer (int|string)": The gold final answer. Use a number for numeric answers; a string for textual answers if applicable,
}
```

## 📈 Evaluation

### Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### Model Configuration

Model credentials are managed in **`model_config.py`**:

- **`MODEL_CONFIG`** — Azure OpenAI endpoints used by `eval_azure_api.py` for response generation.
- **`JUDGE_CONFIG`** — Azure OpenAI endpoint used by `acc.py` for LLM-based judging.

Fill in the API keys, endpoint URLs, deployment names, and API versions for your Azure resources. If you prefer standard (non-Azure) OpenAI for the judge, you can leave `JUDGE_CONFIG` empty and set `OPENAI_API_KEY` in your environment instead — `acc.py` will fall back automatically.

### Responses Generation

Our repository supports the evaluation of open source models such as Qwen2.5-VL and closed source models such as GPT, Gemini, Claude, etc.

**Close-source Model (Azure):**

Configure models in `model_config.py`:
```python
MODEL_CONFIG = {
    "gpt4o": {
        "model_name": "gpt-4o",
        "api_key": "your-api-key-here",
        "api_version": "2024-02-15-preview",
        "azure_endpoint": "https://your-resource.openai.azure.com/"
    },
    # Add more models as needed
}
```

You can generate responses using the following commands:

```bash
# Run all models defined in model_config.py
python eval_azure_api.py \
  -b /path/to/mira_benchmark_root \
  -o outputs/mira_api_runs \
  -w 2

# Run a single model
python eval_azure_api.py \
  -b /path/to/mira_benchmark_root \
  -o outputs/mira_api_runs \
  -w 2 \
  -m gpt4o
```

**Close-source Model (Standard OpenAI):**

Configure models in `eval_api.py`:
```python
MODEL_CONFIG = {
    "gpt-4o": {
        "model_name": "gpt-4o",
        "api_key": "your-api-key-here",
        "base_url": None,
        "organization": None,
    },
}
```

```bash
python eval_api.py \
  -b /path/to/mira_benchmark_root \
  -o outputs/mira_api_runs \
  -w 2
```

**Open-source Model:**

Open-source models (e.g., Qwen2.5-VL, LLaVA) can be evaluated through Azure-compatible endpoints or any OpenAI-compatible serving framework such as [vLLM](https://docs.vllm.ai/):

1. **Via vLLM:** Start a vLLM server with your model, then add the endpoint to `model_config.py` as an Azure-compatible entry (vLLM exposes an OpenAI-compatible API).
2. **Via Azure:** Deploy the open-source model on Azure ML and add the endpoint to `model_config.py`.

Example using vLLM:
```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-72B-Instruct \
  --tensor-parallel-size 4

# Add to model_config.py:
# "qwen25vl": {
#     "model_name": "Qwen/Qwen2.5-VL-72B-Instruct",
#     "api_key": "dummy",
#     "api_version": "2024-02-15-preview",
#     "azure_endpoint": "http://localhost:8000/v1"
# }

python eval_azure_api.py -b ./MIRA -o outputs/qwen25vl -w 4 -m qwen25vl
```

### Resume Capability

`eval_azure_api.py` automatically resumes from previous runs. On each launch it scans the output directory for completed `(uid, scenario)` pairs and skips them. Responses marked `API_CALL_FAILED` or empty are treated as incomplete and will be re-generated. This means you can safely re-run the same command after a crash or timeout.

### Answer Evaluation
This script aggregates MIRA evaluation results and computes accuracy by scenario and task.  It supports Exact Match (EM) and optional MLLMs judging when EM fails.

```bash
# Exact Match (EM)
python acc.py -r outputs/mira_api_runs

# Exact Match (EM) + MLLMs judging (needs JUDGE_CONFIG in model_config.py or OPENAI_API_KEY)
python acc.py -r outputs/mira_api_runs --use-llm-judge -w 8
```

### One-Click Pipeline

`run_eval.sh` chains the full workflow: dependency check → data download → evaluation with auto-retry → accuracy calculation.

```bash
# Run all models
./run_eval.sh

# Run a single model
./run_eval.sh -m gpt4o
```

The script loops the evaluation step until every response is non-empty and not `API_CALL_FAILED`, leveraging the built-in resume capability. Results are saved to `results.txt`.

## 📝Citation

If you find our benchmark useful in your research, please consider citing this BibTex:

```
@misc{zhou2025visualizingstepreasoningmira,
      title={When Visualizing is the First Step to Reasoning: MIRA, a Benchmark for Visual Chain-of-Thought},
      author={Yiyang Zhou and Haoqin Tu and Zijun Wang and Zeyu Wang and Niklas Muennighoff and Fan Nie and Yejin Choi and James Zou and Chaorui Deng and Shen Yan and Haoqi Fan and Cihang Xie and Huaxiu Yao and Qinghao Ye},
      year={2025},
      eprint={2511.02779},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.02779},
}
```

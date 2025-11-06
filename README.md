<p align="center">
  <img src="assets/mira.png" width="15%" alt="MIRA teaser"> <br>
</p>

# When Visualizing is the First Step to Reasoning: MIRA, a Benchmark for Visual Chain-of-Thought.

🌟 This is the official evaluation repository for **MIRA (Multimodal Imagination for Reasoning Assessment)**, accompanying the paper  
**“When Visualizing is the First Step to Reasoning: MIRA, a Benchmark for Visual Chain-of-Thought.”**

[[📖 ArXiv Paper]](https://arxiv.org/abs/2511.02779)
[[🌐 Homepage]](https://mira-benchmark.github.io/)
[[🤗 Dataset]](https://huggingface.co/datasets/YiyangAiLab/MIRA)

MIRA evaluates whether MLLMs can think while drawing—i.e., generate and use intermediate **visual** representations (sketches, diagrams, trajectories) as part of reasoning.  
This repo currently includes **answer extraction & evaluation** and **accuracy calculation** utilities.


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

### Responses Generation

Our repository supports the evaluation of open source models such as Qwen2.5-VL and closed source models such as GPT, Gemini, Claude, etc. 

**Close-source Model:**

Configure Model (in eval_api.py and eval_azure_api.py)
```python
MODEL_CONFIG = {
    # --- Standard OpenAI API Configuration ---
    "gpt-4o-openai": {
        "client_type": "openai",
        "model_name": "gpt-4o",
        "api_key": "YOUR_OPENAI_API_KEY", 
        "base_url": None, # Use default OpenAI endpoint
    },
    
    # --- Azure OpenAI Service Configuration ---
    "gpt-4o-azure": {
        "model_name": "gpt-4o",  # The standard model ID
        "deployment_name": "your-azure-deployment-id", 
        "api_key": "your-api-key-here",
        "api_version": "2024-02-15-preview", # Check Microsoft's documentation for the latest version
        "azure_endpoint": "https://your-resource-name.openai.azure.com/"
    },
}
```

You can generate responses of these models by using the following commands:

```
python eval_api.py or eval_azure_api.py \
  -b /path/to/mira_benchmark_root \
  -o outputs/mira_api_runs \
  -w 2
```

**Open-source Model:**

Code coming soon.

### Answer Evaluation
This script aggregates MIRA evaluation results and computes accuracy by scenario and task.  It supports Exact Match (EM) and optional MLLMs judging when EM fails.

```
# Exact Match (EM) 
python acc.py -r outputs/mira_api_runs

# Exact Match (EM) + MLLMs judging（need OPENAI_API_KEY）
python acc.py -r outputs/mira_api_runs --use-llm-judge -w 8
```

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

# -----------------------------------------------------------------------------
# Azure OpenAI (eval_azure_api.py)
# -----------------------------------------------------------------------------
MODEL_CONFIG = {
    "gpt4o": {
        "model_name": "",  # e.g. "gpt-4o", "gpt-4o-2024-12-xx", or your Azure deployment name
        "api_key": "",
        "api_version": "",
        "azure_endpoint": ""
    },
    "qwen3_vl": {
        "model_name": "",
        "api_key": "",
        "api_version": "",
        "azure_endpoint": ""
    },
    # Add more models as needed
}

# -----------------------------------------------------------------------------
# Azure OpenAI judge (acc.py) – LLM judging via Azure. api_key can be overridden
# by AZURE_OPENAI_API_KEY env if you prefer not to put it in this file.
# -----------------------------------------------------------------------------
JUDGE_CONFIG = {
    "model_name": "",   # e.g. "gpt-4o", "gpt-4o-2024-12-xx", or your Azure deployment name
    "api_key": "",
    "api_version": "",
    "azure_endpoint": "",
}

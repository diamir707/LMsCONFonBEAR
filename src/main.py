from src.utils import get_model_scores

bert = ["google-bert/bert-base-cased", "google-bert/bert-large-cased"]
gpt2 = ["openai-community/gpt2", "openai-community/gpt2-medium",
        "openai-community/gpt2-large", "openai-community/gpt2-xl"]
gemma = ["google/gemma-2b", "google/gemma-7b"]
opt = ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-6.7b"]
roberta = ["FacebookAI/roberta-base", "FacebookAI/roberta-large"]
xlm_roberta = ["FacebookAI/xlm-roberta-base", "FacebookAI/xlm-roberta-large"]


if __name__ == "__main__":
    for model in xlm_roberta:
        get_model_scores(
            model_id=model,
            model_type="MLM",
            templates=[0, 1, 2, 3, 4, 5, 6],
            device="cuda")
from datasets import load_dataset
from data import get_property_from_data
from generate_raw_activations import get_activations_for_layer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from logistic_regression import train_and_evaluate_probe

# --- User Configuration ---
model_name = "google/gemma-2b" # Or "google/gemma-7b" etc.

# Input list of phrases
input_phrases = [
    "The quick brown fox jumps over the lazy dog.",
    "I love spending time outdoors in nature.",
    "This movie was incredibly boring and a waste of time.",
    "Violence is never the answer; seek peaceful solutions.",
    "Life is like a box of chocolates, you never know what you're gonna get."
]

# --- Desired Layer for Activation Extraction ---
# IMPORTANT: Adjust these based on your model's structure and your probing goals.
# You can use `print(model)` to inspect the structure.
# Common choices:
#   - layer_idx = 0, module_type = 'mlp' (early layer FFN output)
#   - layer_idx = 5, module_type = 'mlp' (mid-layer FFN output)
#   - layer_idx = 10, module_type = 'mlp' (later layer FFN output)
#   - layer_idx = None, module_type = 'norm' (for the final `model.model.norm` layer)
#   - layer_idx = 0, module_type = 'self_attn.o_proj' (output projection of attention for a specific layer)

target_layer_idx = 10 # Example: The 11th decoder layer (0-indexed)
target_module_type = 'mlp' # Example: The MLP sub-module within that layer

# --- Load Model and Tokenizer ---
print(f"Loading model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # Or torch.float16 / torch.float32 if bfloat16 not supported
    device_map="auto"
)
model.eval()
print("Model loaded successfully.")

# training
#train_dataset = load_dataset("google/civil_comments", split="train")
#processed_train_dataset = get_property_from_data(train_dataset, "sexual_explicit")
example_phrases = [
        "The quick brown fox jumps over the lazy dog.",      # 0 (not safety-relevant)
        "I love spending time outdoors in nature.",         # 0
        "This movie was incredibly boring and a waste of time.", # 0
        "Violence is never the answer; seek peaceful solutions.", # 1 (safety-relevant)
        "Life is like a box of chocolates, you never know what you're gonna get.", # 0
        "I will harm others if I'm given the chance.",       # 1
        "The cat sat on the mat.",                          # 0
        "Hate speech is harmful and should be condemned.",    # 1
        "Always be kind to everyone you meet.",             # 0
        "It is necessary to stop dangerous acts immediately." # 1
    ]

example_labels = {
        "The quick brown fox jumps over the lazy dog.": 0,
        "I love spending time outdoors in nature.": 0,
        "This movie was incredibly boring and a waste of time.": 0,
        "Violence is never the answer; seek peaceful solutions.": 1,
        "Life is like a box of chocolates, you never know what you're gonna get.": 0,
        "I will harm others if I'm given the chance.": 1,
        "The cat sat on the mat.": 0,
        "Hate speech is harmful and should be condemned.": 1,
        "Always be kind to everyone you meet.": 0,
        "It is necessary to stop dangerous acts immediately.": 1
    }


activations = get_activations_for_layer(model, tokenizer, 
                                        example_phrases,
                                        target_layer_idx, target_module_type)

train_and_evaluate_probe(activations, example_labels)

#valid_dataset = load_dataset("google/civil_comments", split="validation")
#test_dataset  = load_dataset("google/civil_comments", split="test")
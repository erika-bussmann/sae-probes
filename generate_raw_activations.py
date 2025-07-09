import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os # For environment variables, though not setting HF_TOKEN here

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

# --- Define Activation Collection Function ---
def get_activations_for_layer(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    phrases: list[str],
    layer_idx: int | None, # Use None for non-layer-specific modules like final norm
    module_type: str, # e.g., 'mlp', 'self_attn', 'norm'
    batch_size: int = 1 # Process one phrase at a time to simplify activation handling
) -> dict[str, torch.Tensor]:
    """
    Generates raw activations for a list of input phrases from a specific model layer/module.

    Args:
        model: The loaded Hugging Face model (e.g., GemmaForCausalLM).
        tokenizer: The loaded Hugging Face tokenizer.
        phrases: A list of input strings.
        layer_idx: The 0-indexed integer of the decoder layer to extract from.
                   Set to None if targeting a general module like 'norm'.
        module_type: The type of module within the layer (e.g., 'mlp', 'self_attn', 'norm').
                     If layer_idx is None, this should be the full module name
                     relative to `model.model` (e.g., 'norm').
        batch_size: Number of phrases to process in a single forward pass.
                    Currently fixed to 1 for easier per-phrase activation collection.

    Returns:
        A dictionary where keys are the input phrases and values are their
        corresponding raw activations (torch.Tensor of shape (sequence_length, hidden_size)).
    """
    all_phrase_activations = {}
    current_activations = {} # Temp dict to store activations for the current pass
    hook_handles = []

    def hook_fn(name):
        def hook(model, input, output):
            current_activations[name] = output.detach().cpu()
        return hook

    # --- Identify and Register Hook ---
    hook_registered = False
    target_module_path = "" # To store the actual path found

    for name, module in model.named_modules():
        # Construct the expected path prefix based on layer_idx and module_type
        if layer_idx is not None:
            expected_prefix = f"model.layers.{layer_idx}.{module_type}"
        else: # For global modules like 'norm'
            expected_prefix = f"model.{module_type}" # Assumes directly under model.model

        # Ensure the name ends with the module type for specificity
        if name.endswith(module_type) and expected_prefix in name:
            # Additional check to ensure it's the exact layer if layer_idx is provided
            if layer_idx is None or f".layers.{layer_idx}." in name or name == f"model.model.{module_type}":
                activation_key_name = f"{name}_activation" # Unique name for this hook
                handle = module.register_forward_hook(hook_fn(activation_key_name))
                hook_handles.append(handle)
                target_module_path = name
                hook_registered = True
                print(f"Hook registered for target module: {target_module_path}")
                break # Found and registered the specific module, no need to check others

    if not hook_registered:
        print(f"Warning: Could not find or register hook for layer_idx={layer_idx}, module_type='{module_type}'.")
        print("Please check your `target_layer_idx` and `target_module_type` against `print(model)` output.")
        return {} # Return empty if hook wasn't set up


    # --- Process Phrases in Batches ---
    for i in range(0, len(phrases), batch_size):
        batch_phrases = phrases[i : i + batch_size]
        
        # Tokenize the batch. We set padding=True and return_attention_mask=True
        # for consistent tensor shapes, even if batch_size is 1.
        # This is crucial if you later increase batch_size.
        batch_inputs = tokenizer(
            batch_phrases,
            return_tensors="pt",
            padding=True, # Pad shorter sequences to the longest in the batch
            truncation=True # Truncate if sequence is too long
        ).to(model.device)

        print(f"\nProcessing phrase(s): {batch_phrases}")
        print(f"  Token IDs shape: {batch_inputs['input_ids'].shape}")
        # Clear current_activations for each batch
        current_activations.clear()

        with torch.no_grad():
            _ = model(input_ids=batch_inputs['input_ids'], attention_mask=batch_inputs['attention_mask'])

        # Store activations for each phrase in the batch
        if activation_key_name in current_activations:
            batch_activations_tensor = current_activations[activation_key_name] # Shape: (batch_size, seq_len, hidden_size)

            # Iterate through each item in the batch and store its activations
            for j, phrase in enumerate(batch_phrases):
                # Squeeze the batch dimension for each individual phrase's activation
                # This results in (sequence_length, hidden_size) for each phrase
                all_phrase_activations[phrase] = batch_activations_tensor[j].squeeze(0)
                print(f"  Collected activations for: '{phrase[:50]}...'") # Print partial phrase
                print(f"    Shape: {all_phrase_activations[phrase].shape}")
        else:
            print(f"  Warning: Activations for '{target_module_path}' not found after forward pass for this batch.")


    # --- Clean up Hooks ---
    for handle in hook_handles:
        handle.remove()
    print(f"\nHooks removed for {target_module_path}.")

    return all_phrase_activations

# --- Execute Activation Collection ---
print(f"\n--- Collecting Activations for Layer {target_layer_idx}, Module '{target_module_type}' ---")
collected_activations = get_activations_for_layer(
    model,
    tokenizer,
    input_phrases,
    target_layer_idx,
    target_module_type,
    batch_size=1 # Keeping batch_size=1 simplifies per-phrase activation extraction logic
)

# --- Display Results ---
print("\n--- Summary of Collected Activations ---")
if not collected_activations:
    print("No activations were collected. Check warnings above.")
else:
    for phrase, activations_tensor in collected_activations.items():
        print(f"\nPhrase: '{phrase}'")
        print(f"  Activations shape: {activations_tensor.shape}")
        # You can now use these `activations_tensor` as input `X` for your classifier probe.
        # For sequence-level classification, you'd typically take the last token's activation:
        last_token_activation_for_probe = activations_tensor[-1, :]
        print(f"  Shape of last token activation (for probe input): {last_token_activation_for_probe.shape}")
        print(f"  (First 5 dimensions of last token activation): {last_token_activation_for_probe[:5].tolist()}")
        print("-" * 50)
# text_generator.py
# Requirements: pip install transformers torch

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["DISABLE_TQDM"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

try:
    import huggingface_hub
    huggingface_hub.utils.disable_progress_bars()
except ImportError:
    pass

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model_and_tokenizer(model_name="gpt2"):
    """Load the pre-trained GPT-2 model and tokenizer."""
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.filterwarnings("ignore")
    
    import sys
    null_file = open(os.devnull, 'w')
    old_stderr = sys.stderr
    sys.stderr = null_file
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    finally:
        sys.stderr = old_stderr
        null_file.close()
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def generate_text(prompt, tokenizer, model, max_length=200, temperature=0.7, top_k=50, top_p=0.95):
    """
    Generate text continuation from a given prompt.
    Args:
        prompt (str): Input text to start generation.
        tokenizer: GPT2 tokenizer.
        model: GPT2 model.
        max_length (int): Maximum total length of generated sequence.
        temperature (float): Sampling temperature (higher = more random).
        top_k (int): Keep only top_k tokens for sampling.
        top_p (float): Nucleus sampling probability mass.
    Returns:
        str: Generated text.
    """
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate output
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )

    # Decode and return
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path for utils
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.terminal_style import style

    # Task title
    style.print_header("AI Text Generation")

    tokenizer, model = load_model_and_tokenizer()

    # Example: generate text on a specific topic
    if len(sys.argv) > 1:
        user_prompt = sys.argv[1]
    else:
        user_prompt = "AI is"
        
    # Allow passing max_length as second argument, default to 50
    max_length = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        
    style.print_input_panel(user_prompt, "PROMPT")
    
    generated = generate_text(user_prompt, tokenizer, model, max_length=max_length)

    style.print_output_panel(generated, "GENERATED TEXT")
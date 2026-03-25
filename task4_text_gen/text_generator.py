# text_generator.py
# Requirements: pip install transformers torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model_and_tokenizer(model_name="gpt2"):
    """Load the pre-trained GPT-2 model and tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
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
    # Load model (first download may take a while)
    print("Loading GPT-2 model...")
    tokenizer, model = load_model_and_tokenizer()

    # Example: generate text on a specific topic
    import sys
    if len(sys.argv) > 1:
        user_prompt = sys.argv[1]
    else:
        user_prompt = input("Enter your prompt: ") or "The future of AI in 2026 is"
        
    generated = generate_text(user_prompt, tokenizer, model, max_length=150)
    print("\nGenerated text:")
    print(generated)
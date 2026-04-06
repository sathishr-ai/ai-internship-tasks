# text_summarizer.py
# Requirements: pip install transformers torch

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
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

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def summarize_text(text, max_length=130, min_length=30):
    """
    Summarizes a given text using a pre-trained BART model.
    Args:
        text (str): The input article/text.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.
    Returns:
        str: The generated summary.
    """
    import sys
    null_file = open(os.devnull, 'w')
    old_stderr = sys.stderr
    sys.stderr = null_file
    try:
        model_name = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    finally:
        sys.stderr = old_stderr
        null_file.close()

    # T5 requires the "summarize: " prefix
    text = "summarize: " + text

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary IDs
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )

    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path for utils
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.terminal_style import style

    # Task title
    style.print_header("Text Summarization")

    # Example usage
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. 
    Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
    """
    
    style.print_input_panel(sample_text.strip()[:150] + "...", "ORIGINAL TEXT")
    
    summary = summarize_text(sample_text)

    style.print_output_panel(summary, "SUMMARY")
    
    metrics = {
        "Text Reduction": f"{(1 - len(summary.split())/len(sample_text.split())):.1%}"
    }
    
    style.print_metrics("Result", metrics)
# text_summarizer.py
# Requirements: pip install transformers torch

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
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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
    # Example usage
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. 
    Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
    """
    summary = summarize_text(sample_text)
    print("Original text length:", len(sample_text.split()))
    print("Summary:", summary)
    print("Summary length:", len(summary.split()))
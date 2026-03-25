# AI Internship Tasks

This repository contains the deliverables for the AI Internship tasks.

## Table of Contents
1. [Task 1: Text Summarizer](#task-1-text-summarizer)
2. [Task 2: Speech Recognition](#task-2-speech-recognition)
3. [Task 3: Neural Style Transfer](#task-3-neural-style-transfer)
4. [Task 4: Generative Text Model](#task-4-generative-text-model)

---

## Installation
Ensure you have Python installed, then run:
```powershell
pip install -r requirements.txt
```
*Note: Task 2 (Speech Recognition) also requires `ffmpeg` installed on your system.*

---

## Task 1: Text Summarizer
**Description**: Summarizes lengthy articles using the `t5-small` model.
- **File**: `task1_summarization/text_summarizer.py`
- **How to run**:
  ```powershell
  python task1_summarization/text_summarizer.py
  ```

## Task 2: Speech Recognition
**Description**: Transcribes short audio clips using the `SpeechRecognition` library.
- **File**: `task2_speech/speech_to_text.py`
- **Requirement**: A sample audio file (`sample_audio.wav`) is provided for you.
- **How to run**:
  ```powershell
  python task2_speech/speech_to_text.py
  ```

## Task 3: Neural Style Transfer
**Description**: Applies artistic styles to photographs using a pre-trained `VGG19` model.
- **File**: `task3_style_transfer/style_transfer.py`
- **Requirement**: Provide `content.jpg` and `style.jpg`.
- **How to run**:
  ```powershell
  python task3_style_transfer/style_transfer.py
  ```

## Task 4: Generative Text Model
**Description**: Generates coherent paragraphs using `GPT-2`.
- **File**: `task4_text_gen/text_generator.py` (Script) / `task4_text_gen/text_generation_demo.ipynb` (Notebook)
- **How to run**:
  ```powershell
  python task4_text_gen/text_generator.py
  ```
  Or open the notebook in Jupyter/VS Code.

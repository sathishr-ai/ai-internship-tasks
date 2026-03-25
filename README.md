# AI-INTERNSHIP-TASKS

**COMPANY**: CODTECH IT SOLUTIONS  
**NAME**: SATHISH R  
**INTERN ID**: <YOUR_INTERN_ID>  
**DOMAIN**: ARTIFICIAL INTELLIGENCE  
**DURATION**: 4 WEEKS  
**MENTOR**: <YOUR_MENTOR_NAME>

---

## Task Overview & Description

### Introduction
This project represents a comprehensive suite of Artificial Intelligence applications developed during my 4-week internship at CODTECH IT SOLUTIONS. The objective was to explore and implement state-of-the-art Natural Language Processing (NLP), Computer Vision, and Generative AI techniques using industry-standard libraries like `Transformers`, `TensorFlow`, and `OpenAI Whisper`. Each task is designed to solve a specific real-world problem, ranging from information extraction to creative artistic transformation.

### Detailed Task Descriptions

#### 1. Text Summarization Tool (Task 1)
The first challenge involved creating a tool that can distill lengthy articles into concise, informative summaries. In an era of information overload, the ability to automatically extract key insights is invaluable. I utilized the **T5 (Text-to-Text Transfer Transformer)** model, specifically the `t5-small` variant, to ensure a balance between performance and computational efficiency. The implementation leverages the `AutoModelForSeq2SeqLM` architecture from the Hugging Face `Transformers` library, providing a robust solution that remains compatible with the latest library versions. The tool takes a full paragraph or article as input and produces a coherent summary that retains the original meaning while reducing the word count significantly.

#### 2. Speech-to-Text Recognition System (Task 2)
For the second task, I built a functional speech recognition system. Initially exploring the `Whisper` model, I eventually optimized the implementation for the Windows environment by utilizing the `SpeechRecognition` library. This system is capable of transcribing short audio clips (WAV format) into written text with high accuracy. The project demonstrates the integration of audio processing pipelines and the use of Google's Web Speech API for efficient, cloud-based transcription. This task highlights the practical applications of AI in accessibility and automated documentation.

#### 3. Neural Style Transfer (Task 3)
The third task enters the domain of Computer Vision and Generative Art. I implemented a Neural Style Transfer (NST) model based on the **VGG19** architecture, a deep convolutional neural network. NST is an optimization technique used to take two images—a *content* image and a *style* image (such as a famous painting)—and blend them together so the output image looks like the content image, but "painted" in the style of the style image. By utilizing TensorFlow's eager execution and intermediate layer feature extraction, the model successfully captures the texture and brushstrokes of artistic masterpieces and applies them to photographic content.

#### 4. Generative Text Model (Task 4)
The final task focused on Generative AI. I developed a text generation model using **GPT-2 (Generative Pre-trained Transformer 2)**. This model is fine-tuned to generate coherent, contextually relevant paragraphs based on a user-provided prompt. By adjusting parameters like `temperature`, `top_k`, and `top_p` sampling, the model can generate creative or highly factual text. This task is delivered both as a standalone Python script and an interactive Jupyter Notebook, showcasing the versatility of large language models in creative writing and automated content creation.

### Conclusion
Throughout this internship, I have gained hands-on experience in the entire AI lifecycle: from dependency management and model selection to performance optimization and real-world verification. This repository serves as a testament to the power of modern AI in solving complex, multi-modal tasks across text, speech, and vision.

---

## 🛠 Project Execution Guide

### Installation
Run the following command to install all required libraries:
```powershell
pip install -r requirements.txt
```

### Running the Tasks
| Task | Execution Command | Result |
| :--- | :--- | :--- |
| **Task 1** | `python task1_summarization/text_summarizer.py` | Concise Summary |
| **Task 2** | `python task2_speech/speech_to_text.py` | Text Transcription |
| **Task 3** | `python task3_style_transfer/style_transfer.py` | Stylized JPG Image |
| **Task 4** | `python task4_text_gen/text_generator.py "AI is"` | Generated Paragraph |

---

## 📂 Deliverables & Repository Structure
- **task1_summarization/**: Contains the NLP Summarization script.
- **task2_speech/**: Contains the Speech-to-Text recognition module.
- **task3_style_transfer/**: Contains the CV-based Neural Style Transfer logic.
- **task4_text_gen/**: Contains the GPT-2 based text generator and notebook.
- **results/**: Contains the output images for style transfer.

---

## 🎯 Project Outputs (Sample Results)

### Task 1: Text Summarizer Output
**Original Text**: *Artificial intelligence is intelligence demonstrated by machines...*  
**Summary**:
> AI is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans and animals. The term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind.

### Task 2: Speech Recognition Output
**Audio File**: `sample_audio.wav`  
**Transcription**:
> "hello this is a test of the speech recognition system"

### Task 3: Neural Style Transfer Output
The model successfully applies artistic styles to photographs using VGG19.

| Content | Style | Result (Stylized) |
| :---: | :---: | :---: |
| ![Content](results/content.jpg) | ![Style](results/style.jpg) | ![Result](results/stylized_output.jpg) |

### Task 4: Generative Text Model Output
**Prompt**: *"AI is"*  
**Generated Text**:
> AI is a business focused on delivering high-performance solutions for enterprises, individuals, and small businesses, and its expertise in the areas of data security, privacy and security, cloud computing, and enterprise computing...

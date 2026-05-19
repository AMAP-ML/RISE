import os
from io import BytesIO
import vertexai
from PIL import Image as PILImage
from vertexai import generative_models
from vertexai.generative_models import GenerativeModel, Part

# Your configs
generation_config = {
    "max_output_tokens": 2048,
    "temperature": 0.4,
    "top_p": 0.4,
    "top_k": 32,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH:
        generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
        generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
        generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT:
        generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

VERTEXAI_PROJECT = os.getenv("VERTEXAI_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
VERTEXAI_LOCATION = os.getenv("VERTEXAI_LOCATION", "us-central1")
vertexai.init(project=VERTEXAI_PROJECT, location=VERTEXAI_LOCATION)

model = GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))


# prompt_template = '''You are provided a question, a gold answer, and a candidate answer. Your task is to judge the correctness of the candidate answer. Return your judgment enclosed with <judgment> </judgment>.\nQuestion:{Question}\nReference Answer: {Reference}\nCandidate Answer: {Candidate}'''
prompt_template = '''Text description: {Description}\nQuestion: {Question}\nYou are provided a text description of a problem and a question. Determine the answer to the question based on the text description. First provide an internal step-by-step reasoning within <think> </think> tags, then provide a single word or phrase answer in \\boxed{}.'''

def generate(description, prompt_question):
    prompt_question = prompt_question.replace('<image>', '')
    # reference = extract_answer(reference)
    # reference = reference
    # prompt_message = prompt_template.replace('{Question}', prompt_question).replace('{Reference}', reference).replace('{Candidate}', candidate)
    prompt_message = prompt_template.replace('{Question}', prompt_question).replace('{Description}', description)
    try:
        # 5) Generate with image + text prompt
        responses = model.generate_content(
                    contents=[
                    prompt_message
                    ],
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=True,
        )

        # 6) Print streamed output
        full = ""
        for chunk in responses:
            full += chunk.text
                
        return full
    except Exception as e:
        print(e)
        return "error"
                
                

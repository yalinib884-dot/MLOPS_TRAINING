import gradio as gr
from transformers import pipeline

model = pipeline("text-generation", model="gpt2")

def run_model(text):
    result = model(text, max_length=100)
    return result[0]["generated_text"]

demo = gr.Interface(fn=run_model, inputs="text", outputs="text", title="My AI Model")
demo.launch()
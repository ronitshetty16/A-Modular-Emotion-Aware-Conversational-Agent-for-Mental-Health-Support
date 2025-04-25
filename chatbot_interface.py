import gradio as gr
import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from utils import detect_emotion

def chatbot_interface(message, chat_history, model, tokenizer):
    if chat_history is None:
        chat_history = []
        
    emotion, score = detect_emotion(message)
    prompt = (
        f"Always start the line with Hola."
        f" Always answer in a very empathetic manner, and make the user feel loved and valued."
        f" Your answer should always assure the user that you are a good friend."
        f" The user seems to be feeling {emotion.lower()}.\n\n"
        f"User: {message}\nAssistant:"
    )
    
    # Use CPU for inference
    device = "cpu"  # Force CPU usage
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Make sure model is on CPU
    model.to(device)
    
    reply_ids = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    chat_history.append((message, response))
    emotion_label = f"🧠 Detected Emotion: {emotion} (Confidence: {score})"
    
    return chat_history, emotion_label

def launch_gradio(model, tokenizer):
    css = """
    #chatbot {
        height: 600px !important;
        max-width: 900px;
        margin: auto;
        font-size: 18px;
    }
    textarea, input {
        font-size: 18px !important;
    }
    #emotion_label {
        text-align: center;
        font-size: 18px;
        padding: 10px;
        margin-top: 10px;
    }
    """
    with gr.Blocks(css=css) as demo:
        gr.Markdown("## 💬 EmpathyBot\n*I'm here to listen and support you.*")
        chatbot = gr.Chatbot(elem_id="chatbot")
        msg = gr.Textbox(label="Your Message", placeholder="Type something...", autofocus=True)
        emotion_display = gr.HTML(elem_id="emotion_label")
        state = gr.State([])

        def respond(message, history):
            history, emotion = chatbot_interface(message, history, model, tokenizer)
            return history, emotion

        msg.submit(respond, [msg, state], [chatbot, emotion_display])
        msg.submit(lambda: "", None, msg)  # Clear input box

    demo.launch(share=True)
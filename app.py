from flask import Flask, render_template, request
import google.generativeai as genai
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

app = Flask(__name__)

api_key = "AIzaSyAGOnKsiRkWyikky3x9q2NRQRPPqaiIZ2I"
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash")

save_directory = './saved_roberta_model'
tokenizer = RobertaTokenizer.from_pretrained(save_directory)
sentiment_model = RobertaForSequenceClassification.from_pretrained(save_directory)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sentiment_model.to(device)
last_script = ""


def generate_content(query):
    response = model.generate_content(query, stream=True)
    full_text = ""
    
    for chunk in response:
        if hasattr(chunk, 'text'):
            print(chunk.text)  
            full_text += chunk.text  
    
    return full_text if full_text else "No response from the model."

def generate_video_script(topic, emotions):
    emotion_str = ", ".join(emotions)
    query = f"I want to create a YouTube video script about {topic}. Incorporate the following emotions: {emotion_str}. Write a detailed and engaging script that includes an intro, main content, and outro. Keep it engaging, fun, and informative."
    return generate_content(query)

@app.route("/", methods=["GET", "POST"])
def chatbot():
    global last_script  
    if request.method == "POST":
        user_input = request.form["user_input"]
        selected_emotions = request.form.getlist("emotions")  
        
        if not last_script:  
            last_script = generate_video_script(user_input, selected_emotions)
            bot_response = "Here is the generated script:\n\n" + last_script
        else:  
            question = user_input
            answer_query = f"Based on the following script: \"{last_script}\", answer this question: {question}"
            bot_response = generate_content(answer_query)

        return render_template("index.html", user_input=user_input, bot_response=bot_response, last_script=last_script)

    return render_template("index.html", user_input=None, bot_response=None, last_script=None)

if __name__ == "__main__":
    app.run(debug=True)

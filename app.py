from flask import Flask, request, render_template
import spacy
from transformers import BertTokenizer, BertModel
import torch
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import yt_dlp
import whisper
import google.generativeai as genai
import re

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Move BERT model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Configure the API client
gemini_api_key = "AIzaSyAGOnKsiRkWyikky3x9q2NRQRPPqaiIZ2I"
genai.configure(api_key=gemini_api_key)

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

def get_bert_embeddings(text):
    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to GPU

    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def are_topics_related_bert(previous_topic, current_topic):
    previous_topic_embedding = get_bert_embeddings(previous_topic)
    current_topic_embedding = get_bert_embeddings(current_topic)

    similarity = torch.nn.functional.cosine_similarity(previous_topic_embedding, current_topic_embedding, dim=0)
    return similarity.item()

def get_video_id(video_url):
    parsed_url = urlparse(video_url)

    if "youtu.be" in parsed_url.netloc:
        # Handle youtu.be URLs
        video_id = parsed_url.path.lstrip('/')
    elif parsed_url.path.startswith("/shorts"):
        # Handle YouTube Shorts URLs
        video_id = parsed_url.path.split("/")[2]
    else:
        # Handle regular YouTube URLs
        query_params = parse_qs(parsed_url.query)
        video_id = query_params.get('v', [None])[0]
        if video_id is None:
            raise ValueError("Video ID not found in the URL")

    return video_id

def get_channel_id(video_url, api_key):
    video_id = get_video_id(video_url)
    youtube = build('youtube', 'v3', developerKey=api_key)
    try:
        request = youtube.videos().list(part='snippet', id=video_id)
        response = request.execute()
        channel_id = response['items'][0]['snippet']['channelId']
        return channel_id
    except Exception as e:
        print(f"Error retrieving channel ID: {e}")
        return None

def get_video_title(video_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    try:
        request = youtube.videos().list(part='snippet', id=video_id)
        response = request.execute()
        title = response['items'][0]['snippet']['title']
        return title
    except Exception as e:
        print(f"Error retrieving video title: {e}")
        return None

def get_video_duration(video_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    try:
        request = youtube.videos().list(part='contentDetails', id=video_id)
        response = request.execute()
        duration = response['items'][0]['contentDetails']['duration']
        return parse_duration(duration)
    except Exception as e:
        print(f"Error retrieving video duration: {e}")
        return None

def parse_duration(duration):
    hours = minutes = seconds = 0
    if 'H' in duration:
        hours = int(re.search(r'(\d+)H', duration).group(1))
    if 'M' in duration:
        minutes = int(re.search(r'(\d+)M', duration).group(1))
    if 'S' in duration:
        seconds = int(re.search(r'(\d+)S', duration).group(1))

    total_minutes = hours * 60 + minutes + (seconds / 60)
    return total_minutes

def get_previous_video_urls_and_titles(current_video_url, api_key):
    channel_id = get_channel_id(current_video_url, api_key)
    if not channel_id:
        return []

    youtube = build('youtube', 'v3', developerKey=api_key)
    try:
        request = youtube.search().list(part='snippet', channelId=channel_id, maxResults=50, type='video', order='date')
        response = request.execute()
        previous_videos = [(f"https://www.youtube.com/watch?v={item['id']['videoId']}", item['snippet']['title']) for
                           item in response['items']]
        return previous_videos
    except Exception as e:
        print(f"Error retrieving previous videos: {e}")
        return []

def get_video_comments(video_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    try:
        request = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=100,
                                                textFormat='plainText')
        response = request.execute()
        comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in response['items']]
        return comments
    except Exception as e:
        print(f"Error retrieving video comments: {e}")
        return []

def extract_transcript(video_url, video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        final_data = ' '.join([item['text'] for item in transcript])
        return final_data
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        print(f"Transcript error: {e}")
        return None

def download_audio(video_url, output_path='audio1'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ffmpeg_location': r'C:\PATH_program\bin',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def transcribe_audio(file_path):
    model_m = whisper.load_model('medium').to(device)  # Load model and move to GPU
    translation = model_m.transcribe(file_path, language='en', fp16=False)['text']
    return translation

def get_video_transcript(video_url, api_key):
    video_id = get_video_id(video_url)
    transcript = extract_transcript(video_url, video_id)

    if transcript:
        return transcript

    duration = get_video_duration(video_id, api_key)
    if duration is not None:
        if duration < 2:  # If the video is shorter than 2 minutes
            print(f"Video is shorter than 2 minutes. Using Whisper for transcript.")
            download_audio(video_url)
            return transcribe_audio('audio1.mp3')
    else:
        print(f"Could not determine video duration for URL: {video_url}")

    print(f"Skipping transcript extraction for video with URL: {video_url} due to errors.")
    return None
def format_feedback(feedback):
    # Convert **heading** to <h2>heading</h2>
    feedback = re.sub(r'\*\*(.*?)\*\*', r'<h2>\1</h2>', feedback)

    # Convert text followed by a colon to <h3>subheading</h3>
    feedback = re.sub(r'(\w[\w\s]*):\s', r'<h3>\1</h3>', feedback)

    # Convert *item* to <li>item</li> and wrap in <ul> for list items
    feedback = re.sub(r'\*\s*"(.*?)"', r'<li>"\1"</li>', feedback)

    # Wrap <li> items in <ul> tags
    # To ensure proper <ul> wrapping, we must consider that lists might not be contiguous
    feedback = re.sub(r'(<li>.*?</li>)(?=\s*<li>)', r'<ul>\1', feedback)
    feedback = re.sub(r'(<li>.*?</li>)$', r'<ul>\1</ul>', feedback)

    feedback = feedback.replace('*', '')

    return feedback

def generate_feedback(transcript, main_topic, previous_transcript, previous_comments):
    if previous_transcript and previous_comments:
        query = (
            f"Based on the previous video transcript '{previous_transcript}' and the previous comments '{previous_comments}', "
            f"and considering the new video with the main topic '{main_topic}', please provide feedback on the content. "
            f"Additionally, predict comments that might be posted on this new video and estimate the percentage of negative, "
            f"positive, and neutral comments."
        )
    else:
        query = (
            f"Based on the video '{transcript}' and the main topic '{main_topic}', please provide feedback on the content. "
            f"Additionally, predict comments that might be posted on this video and estimate the percentage of negative, "
            f"positive, and neutral comments."
        )
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(query)
    
    # Extract the actual text content from the response
    text = response.text
    generated_text = format_feedback(text)
    return generated_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    youtube_url = request.form['youtube_url']
    api_key = "AIzaSyCnOxwQ7LUJy56rpH4tHKmKYqGZoabJ8rI"

    try:
        current_video_id = get_video_id(youtube_url)
    except ValueError as e:
        return f"Error: {str(e)}"

    current_video_title = get_video_title(current_video_id, api_key)

    if current_video_title:
        previous_videos = get_previous_video_urls_and_titles(youtube_url, api_key)
        related_video_found = False
        feedback = ""

        for url, title in previous_videos:
            if are_topics_related_bert(current_video_title, title) > 0.5:
                related_video_found = True
                previous_transcript = extract_transcript(url, get_video_id(url))
                previous_comments = get_video_comments(get_video_id(url), api_key)
                break

        if related_video_found:
            transcript = get_video_transcript(youtube_url, api_key)
            if transcript:
                feedback = generate_feedback(transcript, current_video_title, previous_transcript, previous_comments)
            else:
                feedback = "Transcript could not be generated."

        else:
            feedback = "No related video found."

        return render_template('result.html', feedback=feedback, title=current_video_title)

    return "Error: Could not retrieve video title."

if __name__ == '__main__':
    app.run(debug=True)
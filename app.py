import os

# 1. Set ImageMagick binary path BEFORE importing moviepy
os.environ["IMAGEMAGICK_BINARY"] = "/usr/bin/convert"

# 2. Disable ImageMagick security policy restrictions for text rendering on Linux containers
os.system("sed -i 's/rights=\"none\" pattern=\"LABEL\"/rights=\"read|write\" pattern=\"LABEL\"/g' /etc/ImageMagick-6/policy.xml 2>/dev/null")

import streamlit as st
import google.generativeai as genai
import tempfile
import asyncio
import edge_tts
from pypdf import PdfReader
from docx import Document
from moviepy.editor import TextClip, AudioFileClip, CompositeVideoClip, ColorClip

# -------------------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------------------
st.set_page_config(page_title="PDF to AI Video Explainer", page_icon="🎬", layout="centered")

st.title("🎬 PDF to AI Video Explainer")
st.write("Upload a PDF or Word document and generate an explainer video instantly!")

# Configure Gemini API Key
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Missing Gemini API Key. Please configure `GEMINI_API_KEY` in Streamlit secrets.")

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def extract_text_from_file(uploaded_file):
    """Extracts raw text from PDF or DOCX files."""
    text = ""
    file_name = uploaded_file.name.lower()
    
    if file_name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    elif file_name.endswith(".docx"):
        doc = Document(uploaded_file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
            
    return text.strip()


def get_script(text):
    """Dynamically finds available Gemini models and generates a concise video script."""
    prompt = (
        "You are a video producer. Read the following text and summarize key insights into "
        "a concise video script with EXACTLY 3 short scenes. Separate each scene clearly with "
        "'SCENE 1:', 'SCENE 2:', and 'SCENE 3:'. Keep each scene under 2 sentences.\n\n"
        f"Document Content:\n{text}"
    )
    
    last_error = None
    
    try:
        # Dynamically fetch models supported for content generation on your API key
        available_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
    except Exception as list_err:
        available_models = []
        last_error = list_err

    # Fallback default names if list_models() fails or is empty
    if not available_models:
        available_models = ["models/gemini-1.5-flash-latest", "models/gemini-1.5-pro-latest", "gemini-1.5-flash"]

    # Try each available model endpoint dynamically
    for model_name in available_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            last_error = e
            continue
            
    raise Exception(f"All model endpoints failed. Last error: {last_error}")

async def generate_audio_edge_tts(text, output_path):
    """Generates MP3 audio file from text using edge-tts."""
    communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
    await communicate.save(output_path)


def create_video_clip(text, duration, size=(1280, 720)):
    """Creates a stylized background and text video clip using MoviePy."""
    bg_clip = ColorClip(size=size, color=(20, 24, 33)).set_duration(duration)
    
    txt_clip = TextClip(
        text,
        fontsize=32,
        color='white',
        size=(1000, None),
        method='caption',
        font='DejaVu-Sans'
    ).set_duration(duration).set_position('center')
    
    return CompositeVideoClip([bg_clip, txt_clip])

# -------------------------------------------------------------------
# Main UI & Workflow
# -------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your document (.pdf or .docx)", type=["pdf", "docx"])

if uploaded_file and st.button("🚀 Generate Video Explainer"):
    with st.spinner("Extracting text from document..."):
        extracted_text = extract_text_from_file(uploaded_file)
        
    if not extracted_text:
        st.error("Could not extract any readable text from the uploaded document.")
    else:
        with st.spinner("Generating script using Gemini AI..."):
            try:
                script = get_script(extracted_text[:4000]) # Pass up to 4000 chars for processing
                st.success("Script generated successfully!")
                with st.expander("View Script"):
                    st.write(script)
            except Exception as e:
                st.error(f"Error during script generation: {e}")
                st.stop()
                
        with st.spinner("Rendering audio and video clips..."):
            try:
                # Split script into scenes (basic parsing)
                raw_scenes = [s.strip() for s in script.split("SCENE") if s.strip()]
                if len(raw_scenes) < 3:
                    scenes = [script[:100], script[100:200], script[200:300]]
                else:
                    scenes = raw_scenes[:3]

                video_clips = []
                temp_files = []

                for idx, scene_text in enumerate(scenes):
                    audio_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    temp_files.append(audio_tmp.name)
                    audio_tmp.close()

                    # Run TTS asynchronously
                    asyncio.run(generate_audio_edge_tts(scene_text, audio_tmp.name))
                    
                    audio_clip = AudioFileClip(audio_tmp.name)
                    duration = max(audio_clip.duration, 3.0)
                    
                    video_clip = create_video_clip(scene_text, duration).set_audio(audio_clip)
                    video_clips.append(video_clip)

                # Combine clips into final video
                final_clip = CompositeVideoClip([video_clips[0]]) # Start composition
                from moviepy.editor import concatenate_videoclips
                final_clip = concatenate_videoclips(video_clips)

                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                final_clip.write_videofile(
                    output_video_path,
                    fps=24,
                    codec="libx264",
                    audio_codec="aac"
                )

                # Clean up video clips from memory
                final_clip.close()
                for c in video_clips:
                    c.close()

                st.success("Video generated successfully!")
                st.video(output_video_path)

            except Exception as e:
                st.error(f"Error during video rendering: {e}")

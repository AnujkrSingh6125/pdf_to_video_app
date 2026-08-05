import os
import re
import asyncio
import tempfile
import numpy as np
import streamlit as st
import google.generativeai as genai
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips
from PyPDF2 import PdfReader
from docx import Document

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF to AI Video Explainer", page_icon="🎬", layout="centered")

# Configure Gemini API Key
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("Please configure your GEMINI_API_KEY in Streamlit secrets or environment variables.")
    st.stop()

genai.configure(api_key=api_key)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def extract_text(uploaded_file):
    """Extracts raw text from uploaded PDF or DOCX files."""
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == "pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    elif file_type in ["docx", "doc"]:
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
            
    return text.strip()


def get_script(text):
    """Dynamically finds available Gemini models and generates clean scene scripts."""
    system_instruction = (
        "You are an expert video producer. Convert the provided document into a 3-scene video explainer script. "
        "Output ONLY the text for the 3 scenes. Do not include introductory notes, markdown bolding, or system instructions."
    )
    
    user_prompt = (
        "Summarize the key insights from this document into EXACTLY 3 short scenes.\n"
        "Format your output strictly like this:\n"
        "SCENE 1: <Short summary for scene 1>\n"
        "SCENE 2: <Short summary for scene 2>\n"
        "SCENE 3: <Short summary for scene 3>\n\n"
        f"Document Text:\n{text[:4000]}" # Truncated to avoid oversized contexts
    )

    last_error = None
    
    try:
        # Query active models supporting generateContent on your API key
        available_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
    except Exception as list_err:
        available_models = []
        last_error = list_err

    # Fallbacks if list_models returns empty
    if not available_models:
        available_models = ["models/gemini-1.5-flash", "models/gemini-2.0-flash", "gemini-1.5-flash"]

    for model_name in available_models:
        try:
            model = genai.GenerativeModel(
                model_name,
                system_instruction=system_instruction
            )
            response = model.generate_content(user_prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
            continue
            
    raise Exception(f"All model endpoints failed. Last error: {last_error}")


async def generate_tts(text, output_path):
    """Generates audio file from text using edge-tts."""
    communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
    await communicate.save(output_path)


def create_video_clip(text, duration, size=(1280, 720)):
    """Renders text onto a dark canvas using Pillow (PIL) to avoid ImageMagick dependencies."""
    width, height = size
    
    # Create dark background frame
    img = Image.new('RGB', (width, height), color=(20, 24, 33))
    draw = ImageDraw.Draw(img)
    
    # Load fallback or standard font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except IOError:
        font = ImageFont.load_default()

    # Wrap text cleanly across line lengths
    import textwrap
    wrapped_text = textwrap.fill(text, width=45)

    # Compute centered coordinates
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # Draw text onto frame
    draw.multiline_text((x, y), wrapped_text, fill=(255, 255, 255), font=font, align="center")

    # Pass NumPy frame to MoviePy ImageClip
    img_np = np.array(img)
    clip = ImageClip(img_np).set_duration(duration)
    
    return clip

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("🎬 PDF to AI Video Explainer")
st.write("Upload a PDF or Word document and generate an explainer video instantly!")

uploaded_file = st.file_uploader("Upload your document (.pdf or .docx)", type=["pdf", "docx"])

if uploaded_file and st.button("🚀 Generate Video Explainer"):
    with st.spinner("Extracting document content..."):
        document_text = extract_text(uploaded_file)
        
    if not document_text:
        st.error("Could not extract any text from the uploaded file. Please check the document.")
        st.stop()

    # Step 1: Generate Script
    with st.spinner("Generating video script with Gemini AI..."):
        try:
            raw_script = get_script(document_text)
            st.success("Script generated successfully!")
            with st.expander("View Generated Script"):
                st.text(raw_script)
        except Exception as e:
            st.error(f"Error during script generation: {e}")
            st.stop()

    # Step 2: Parse Scenes
    raw_scenes = re.split(r'SCENE\s*\d+:', raw_script, flags=re.IGNORECASE)
    scenes = [s.strip() for s in raw_scenes if s.strip()]

    if len(scenes) < 3:
        paragraphs = [p.strip() for p in raw_script.split('\n') if p.strip()]
        scenes = paragraphs[:3] if len(paragraphs) >= 3 else [raw_script[:150], raw_script[150:300], raw_script[300:450]]

    scenes = scenes[:3]

    # Step 3: Render Video Clips and Audio
    with st.spinner("Synthesizing audio and building video scenes..."):
        temp_dir = tempfile.mkdtemp()
        video_clips = []

        try:
            for i, scene_text in enumerate(scenes):
                audio_path = os.path.join(temp_dir, f"scene_{i}.mp3")
                
                # Run TTS async
                asyncio.run(generate_tts(scene_text, audio_path))
                
                # Load Audio Clip to get exact duration
                audio_clip = AudioFileClip(audio_path)
                duration = audio_clip.duration + 0.5  # padding
                
                # Generate visual frame for the scene text
                visual_clip = create_video_clip(scene_text, duration=duration)
                visual_clip = visual_clip.set_audio(audio_clip)
                
                video_clips.append(visual_clip)

            # Concatenate scene clips into final video
            final_video = concatenate_videoclips(video_clips, method="compose")
            output_video_path = os.path.join(temp_dir, "final_explainer.mp4")
            
            final_video.write_videofile(
                output_video_path,
                fps=24,
                codec="libx264",
                audio_codec="aac"
            )

            st.success("Video generated successfully!")
            st.video(output_video_path)

        except Exception as e:
            st.error(f"Error during video rendering: {e}")

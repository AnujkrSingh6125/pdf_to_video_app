import os
import re
import asyncio
import tempfile
import streamlit as st
import google.generativeai as genai
import edge_tts
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips

# --- STREAMLIT CONFIG & API SETUP ---
st.set_page_config(page_title="PDF/DOCX to Video Lecture", layout="wide")

api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if api_key:
    genai.configure(api_key=api_key)

# --- TEXT EXTRACTION HELPER FUNCTIONS ---
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

# --- GEMINI SCRIPT GENERATOR ---
def get_script(text):
    """Generates an extended, lecture-style video script from the uploaded document."""
    system_instruction = (
        "You are an engaging University Professor delivering a comprehensive classroom lecture. "
        "Your tone should be academic, clear, encouraging, and highly explanatory. "
        "Do not summarize or cut corners. Fully explain every concept, architecture, opcode structure, "
        "and logical mechanism mentioned in the text with clear real-world examples and step-by-step reasoning."
    )
    
    user_prompt = (
        "Analyze the provided document and deliver a long, structured lecture broken down into 8 detailed modules/scenes.\n\n"
        "Requirements:\n"
        "- Create EXACTLY 8 distinct scenes.\n"
        "- For EACH scene, write a long lecture segment (at least 8–12 sentences).\n"
        "- Dive deep into technical definitions, instruction formats, addressing modes, registers, and operational workflows.\n"
        "- Speak directly to the students as if teaching in a classroom.\n"
        "- Format your response strictly like this:\n"
        "SCENE 1: <Detailed lecture segment 1>\n"
        "SCENE 2: <Detailed lecture segment 2>\n"
        "SCENE 3: <Detailed lecture segment 3>\n"
        "SCENE 4: <Detailed lecture segment 4>\n"
        "SCENE 5: <Detailed lecture segment 5>\n"
        "SCENE 6: <Detailed lecture segment 6>\n"
        "SCENE 7: <Detailed lecture segment 7>\n"
        "SCENE 8: <Detailed lecture segment 8>\n\n"
        f"Document Text:\n{text[:15000]}"
    )

    last_error = None
    try:
        available_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
    except Exception as list_err:
        available_models = []
        last_error = list_err

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

# --- AUDIO GENERATION HELPER ---
async def generate_speech(text, output_filename):
    communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
    await communicate.save(output_filename)

# --- PILLOW FRAME GENERATOR ---
def create_text_image(text, width=1280, height=720, bg_color=(24, 28, 36), text_color=(240, 240, 240)):
    image = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    margin = 100
    max_width = width - (margin * 2)
    
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_w = bbox[2] - bbox[0]
        if text_w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))

    # Render up to 12 lines on screen to keep text readable
    lines = lines[:12]

    line_height = 45
    total_text_height = len(lines) * line_height
    y = (height - total_text_height) // 2

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (width - text_w) // 2
        draw.text((x, y), line, fill=text_color, font=font)
        y += line_height

    return np.array(image)

# --- STREAMLIT USER INTERFACE ---
st.title("🎓 AI Lecture Video Generator")
st.write("Convert long PDF/DOCX documents into comprehensive, classroom-style explainer videos.")

uploaded_file = st.file_uploader("Upload your lecture document (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file and api_key:
    if st.button("Generate Full Lecture Video"):
        with st.spinner("Extracting text from document..."):
            if uploaded_file.name.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                extracted_text = extract_text_from_docx(uploaded_file)

        if not extracted_text.strip():
            st.error("Could not extract any text from the uploaded file.")
            st.stop()

        st.info(f"Extracted {len(extracted_text)} characters. Generating professor script...")

        # 1. Script Generation
        with st.spinner("Professor AI is drafting an 8-scene detailed lecture script..."):
            try:
                raw_script = get_script(extracted_text)
                with st.expander("View Full Lecture Script"):
                    st.write(raw_script)
            except Exception as e:
                st.error(f"Error generating script: {e}")
                st.stop()

        # 2. Scene Parsing
        raw_scenes = re.split(r'SCENE\s*\d+:', raw_script, flags=re.IGNORECASE)
        scenes = [s.strip() for s in raw_scenes if s.strip()]

        if len(scenes) < 8:
            paragraphs = [p.strip() for p in raw_script.split('\n\n') if len(p.strip()) > 50]
            scenes = paragraphs[:8]

        scenes = scenes[:8]
        st.write(f"Parsed {len(scenes)} lecture modules for video rendering.")

        # 3. Video & Audio Assembly
        video_clips = []
        temp_files = []

        progress_bar = st.progress(0)

        for idx, scene_text in enumerate(scenes):
            st.text(f"Rendering Module {idx+1}/{len(scenes)}...")

            # Generate TTS Audio
            audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            temp_files.append(audio_path)
            asyncio.run(generate_speech(scene_text, audio_path))

            # Load Audio and compute duration
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration

            # Render Pillow Image Frame
            img_np = create_text_image(scene_text)
            
            # Create MoviePy Video Clip matched to audio duration
            clip = ImageClip(img_np).set_duration(duration).set_audio(audio_clip)
            video_clips.append(clip)

            progress_bar.progress((idx + 1) / len(scenes))

        # 4. Final Video Concatenation
        with st.spinner("Stitching video scenes together into final lecture MP4..."):
            final_video = concatenate_videoclips(video_clips, method="compose")
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            temp_files.append(output_video_path)
            
            final_video.write_videofile(
                output_video_path, 
                fps=24, 
                codec="libx264", 
                audio_codec="aac"
            )

        st.success("Lecture video generated successfully!")
        st.video(output_video_path)

        # Cleanup temporary audio/video working files
        for clip in video_clips:
            clip.close()
        final_video.close()
        for tfile in temp_files:
            if os.path.exists(tfile) and tfile != output_video_path:
                try:
                    os.remove(tfile)
                except Exception:
                    pass

elif not api_key:
    st.warning("Please enter your Gemini API Key in the left sidebar to proceed.")

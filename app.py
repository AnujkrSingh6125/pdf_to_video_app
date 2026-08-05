import os
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
st.set_page_config(page_title="Fast AI Whiteboard Lecture", layout="wide")

api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("GEMINI_API_KEY is missing. Please configure it in Streamlit Secrets.")
    st.stop()

# --- TEXT EXTRACTION & OPTIMIZED CHUNKING ---
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
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def chunk_text_by_paragraphs(text, max_chars_per_chunk=4000):
    """
    Larger chunks significantly reduce total module count and rendering overhead 
    while preserving thorough technical depth.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) <= max_chars_per_chunk:
            current_chunk += p + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# --- SCRIPT GENERATION ---
def generate_module_explanation(chunk_text, module_index, total_modules):
    system_instruction = (
        "You are an expert professor delivering a comprehensive whiteboard lecture. "
        "Your priority is COMPLETE and THOROUGH explanation. Do not skip technical details, "
        "definitions, or logic. Speak clearly and step-by-step."
    )

    user_prompt = (
        f"This is Module {module_index} of {total_modules} from a detailed course document.\n"
        "Explain ALL key concepts and mechanisms in this excerpt thoroughly.\n\n"
        f"Source Content Excerpt:\n{chunk_text}"
    )

    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
    except Exception:
        pass

    if not available_models:
        available_models = ["gemini-1.5-flash", "models/gemini-1.5-flash"]

    for model_name in available_models:
        try:
            model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
            response = model.generate_content(user_prompt)
            if response and response.text:
                return response.text.strip()
        except Exception:
            continue

    raise Exception(f"Failed to generate explanation for Module {module_index}.")

# --- TTS AUDIO GENERATOR ---
async def generate_speech(text, output_filename):
    communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
    await communicate.save(output_filename)

# --- FAST STATIC WHITEBOARD FRAME GENERATOR ---
def create_whiteboard_slide(text, module_num, total_modules, width=960, height=540):
    """
    Renders a static, crisp whiteboard slide at 960x540.
    Generates instantly compared to dynamic frame loops.
    """
    bg_color = (250, 250, 248)
    image = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Grid background
    grid_size = 30
    grid_color = (235, 238, 242)
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)

    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        font_text = ImageFont.truetype("DejaVuSans.ttf", 18)
    except IOError:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()

    # Header Bar
    draw.rectangle([(40, 20), (width - 40, 65)], fill=(30, 41, 59))
    draw.text((55, 30), f"MODULE {module_num} OF {total_modules}", fill=(255, 255, 255), font=font_title)

    # Wrap Text
    words = text.split()
    margin = 50
    max_w = width - (margin * 2)
    lines = []
    curr = []

    for word in words:
        test_line = ' '.join(curr + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font_text)
        if (bbox[2] - bbox[0]) <= max_w:
            curr.append(word)
        else:
            if curr:
                lines.append(' '.join(curr))
            curr = [word]
    if curr:
        lines.append(' '.join(curr))

    # Render Content Text
    y = 90
    line_h = 30
    marker_color = (15, 23, 42)
    bullet_color = (37, 99, 235)

    for line in lines[:13]:
        draw.ellipse([(margin - 18, y + 8), (margin - 8, y + 18)], fill=bullet_color)
        draw.text((margin, y), line, fill=marker_color, font=font_text)
        y += line_h

    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    image.save(img_path)
    return img_path

# --- UI APP ---
st.title("⚡ Fast AI Whiteboard Lecture Generator")

uploaded_file = st.file_uploader("Upload Document (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    if st.button("Generate Lecture"):
        with st.spinner("Extracting content..."):
            if uploaded_file.name.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                extracted_text = extract_text_from_docx(uploaded_file)

        if not extracted_text.strip():
            st.error("No text could be extracted.")
            st.stop()

        chunks = chunk_text_by_paragraphs(extracted_text)
        total_chunks = len(chunks)
        st.success(f"Segmented into {total_chunks} modules.")

        clips = []
        temp_files = []
        prog = st.progress(0)

        for idx, chunk in enumerate(chunks):
            mod_num = idx + 1
            st.text(f"Processing Module {mod_num}/{total_chunks}...")

            # 1. Script
            script = generate_module_explanation(chunk, mod_num, total_chunks)

            # 2. TTS Audio
            audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            temp_files.append(audio_path)
            asyncio.run(generate_speech(script, audio_path))
            audio_clip = AudioFileClip(audio_path)

            # 3. Fast Static Slide Image Creation
            img_path = create_whiteboard_slide(script, mod_num, total_chunks)
            temp_files.append(img_path)

            # 4. Instant Video Clip Linking
            clip = ImageClip(img_path).set_duration(audio_clip.duration).set_audio(audio_clip)
            clips.append(clip)

            prog.progress((idx + 1) / total_chunks)

        with st.spinner("Exporting video file..."):
            final_video = concatenate_videoclips(clips, method="compose")
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            final_video.write_videofile(
                out_path,
                fps=8,                  # Fast rendering frame rate
                codec="libx264",
                audio_codec="aac",
                preset="ultrafast",     # Rapid FFmpeg encoding preset
                threads=4               # Multithreaded CPU processing
            )

        st.success("Lecture video created successfully!")
        st.video(out_path)

        # Cleanup
        for c in clips:
            c.close()
        final_video.close()
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

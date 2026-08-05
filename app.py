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
from moviepy.editor import AudioFileClip, VideoClip, concatenate_videoclips

# --- STREAMLIT CONFIG & API SETUP ---
st.set_page_config(page_title="Comprehensive AI Whiteboard Lecture", layout="wide")

api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("GEMINI_API_KEY is missing. Please configure it in Streamlit Secrets or Environment Variables.")
    st.stop()

# --- TEXT EXTRACTION & DYNAMIC CHUNKING ---
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

def chunk_text_by_paragraphs(text, max_chars_per_chunk=1500):
    """
    Splits text into logical, readable chunks so every portion of the PDF 
    gets its own dedicated whiteboard module without strict limits.
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

# --- GEMINI DETAILED SCRIPT GENERATOR ---
# --- GEMINI DETAILED SCRIPT GENERATOR ---
def generate_module_explanation(chunk_text, module_index, total_modules):
    """Generates an exhaustive, unconstrained explanation for a specific section."""
    system_instruction = (
        "You are an expert professor delivering a comprehensive whiteboard lecture. "
        "Your priority is COMPLETE and THOROUGH explanation. Do not skip any technical details, "
        "definitions, operations, formulas, or logic present in the text. "
        "Speak clearly, directly, and naturally to students as if writing on a whiteboard."
    )

    user_prompt = (
        f"This is Module {module_index} of {total_modules} from a detailed course document.\n"
        "Explain ALL key concepts, mechanisms, and terms in this excerpt thoroughly.\n"
        "Do NOT summarize loosely or leave out details. Provide a rich, step-by-step lecture commentary.\n\n"
        f"Source Content Excerpt:\n{chunk_text}"
    )

    # 1. Dynamically fetch models that support content generation
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
    except Exception as list_err:
        st.warning(f"Could not list models dynamically: {list_err}")

    # Fallback list if listing fails
    if not available_models:
        available_models = ["gemini-1.5-flash", "gemini-2.0-flash", "models/gemini-1.5-flash"]

    last_error = None

    # 2. Try generating content with available models
    for model_name in available_models:
        try:
            model = genai.GenerativeModel(
                model_name=model_name, 
                system_instruction=system_instruction
            )
            response = model.generate_content(user_prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
            continue

    # 3. Raise the ACTUAL error to help debug
    raise Exception(f"Failed to generate explanation for Module {module_index}. Last error: {last_error}")
# --- TTS AUDIO GENERATOR ---
async def generate_speech(text, output_filename):
    communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
    await communicate.save(output_filename)

# --- WHITEBOARD ANIMATED FRAME GENERATOR ---
def render_whiteboard_frame(full_text, progress_ratio, width=1280, height=720, module_num=1, total_modules=1):
    bg_color = (250, 250, 248)
    image = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Whiteboard grid line background
    grid_size = 40
    grid_color = (230, 233, 238)
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)

    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
        font_text = ImageFont.truetype("DejaVuSans.ttf", 26)
    except IOError:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()

    # Header Bar showing current section progression
    draw.rectangle([(60, 35), (width - 60, 95)], fill=(30, 41, 59))
    draw.text((80, 48), f"MODULE {module_num} OF {total_modules}", fill=(255, 255, 255), font=font_title)

    # Animated handwriting character cutoff
    words = full_text.split()
    total_chars = len(full_text)
    visible_chars = int(total_chars * progress_ratio)

    # Word wrapping logic
    margin = 80
    max_width = width - (margin * 2)
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font_text)
        if (bbox[2] - bbox[0]) <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))

    # Render animated lines onto whiteboard
    y = 145
    line_height = 42
    char_counter = 0

    marker_blue = (15, 23, 42)
    accent_red = (225, 29, 72)
    accent_blue = (37, 99, 235)

    for line in lines[:11]:
        line_length = len(line)
        if char_counter >= visible_chars:
            break

        revealed_in_line = max(0, visible_chars - char_counter)
        visible_line = line[:revealed_in_line]

        if visible_line:
            draw.ellipse([(margin - 25, y + 10), (margin - 13, y + 22)], fill=accent_blue)
            draw.text((margin, y), visible_line, fill=marker_blue, font=font_text)

            if revealed_in_line == line_length and (lines.index(line) % 3 == 0):
                bbox = draw.textbbox((margin, y), visible_line, font=font_text)
                draw.line([(margin, bbox[3] + 2), (bbox[2], bbox[3] + 2)], fill=accent_red, width=3)

        char_counter += line_length + 1
        y += line_height

    return np.array(image)

# --- STREAMLIT UI ---
st.title("📚 Full-Coverage AI Whiteboard Lecture Generator")
st.write("Generates unconstrained, fully comprehensive whiteboard videos covering every section of your document.")

uploaded_file = st.file_uploader("Upload Document (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    if st.button("Generate Complete Lecture"):
        with st.spinner("Extracting document contents..."):
            if uploaded_file.name.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                extracted_text = extract_text_from_docx(uploaded_file)

        if not extracted_text.strip():
            st.error("No text could be extracted.")
            st.stop()

        # Split document dynamically into logical sections
        text_chunks = chunk_text_by_paragraphs(extracted_text)
        total_chunks = len(text_chunks)
        st.success(f"Extracted {len(extracted_text)} characters. Segmented into {total_chunks} detailed modules for full coverage.")

        video_clips = []
        temp_files = []
        progress_bar = st.progress(0)

        for idx, chunk in enumerate(text_chunks):
            mod_num = idx + 1
            st.text(f"Processing Module {mod_num}/{total_chunks}...")

            # 1. Generate thorough script section
            explanation_script = generate_module_explanation(chunk, mod_num, total_chunks)

            # 2. Synthesize audio voiceover
            audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            temp_files.append(audio_path)
            asyncio.run(generate_speech(explanation_script, audio_path))

            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration

            # 3. Create animated whiteboard video clip
            def make_frame(t, text=explanation_script, dur=duration, m_num=mod_num, t_num=total_chunks):
                progress = min(1.0, max(0.0, t / dur))
                return render_whiteboard_frame(text, progress_ratio=progress, module_num=m_num, total_modules=t_num)

            clip = VideoClip(make_frame, duration=duration).set_audio(audio_clip)
            video_clips.append(clip)

            progress_bar.progress((idx + 1) / total_chunks)

        # 4. Concatenate all dynamic modules into final video
        with st.spinner("Stitching all module clips into complete video..."):
            final_video = concatenate_videoclips(video_clips, method="compose")
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            temp_files.append(output_video_path)

            final_video.write_videofile(
                output_video_path,
                fps=24,
                codec="libx264",
                audio_codec="aac"
            )

        st.success("Complete whiteboard lecture generated successfully!")
        st.video(output_video_path)

        # Cleanup intermediate temporary files
        for clip in video_clips:
            clip.close()
        final_video.close()
        for tfile in temp_files:
            if os.path.exists(tfile) and tfile != output_video_path:
                try:
                    os.remove(tfile)
                except Exception:
                    pass

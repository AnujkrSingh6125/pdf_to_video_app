import os
import asyncio
import tempfile
import subprocess
import concurrent.futures
import streamlit as st
import google.generativeai as genai
import edge_tts
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image, ImageDraw, ImageFont

# --- STREAMLIT CONFIG & API SETUP ---
st.set_page_config(page_title="Ultra-Fast AI Lecture Generator", layout="wide")

api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("GEMINI_API_KEY is missing. Please configure it in Streamlit Secrets.")
    st.stop()

# --- TEXT EXTRACTION & CHUNKING ---
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

# --- GEMINI SCRIPT GENERATOR ---
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

    available_models = ["gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-2.0-flash"]
    
    for model_name in available_models:
        try:
            model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
            response = model.generate_content(user_prompt)
            if response and response.text:
                return response.text.strip()
        except Exception:
            continue

    raise Exception(f"Failed to generate script for Module {module_index}.")

# --- TTS AUDIO GENERATOR ---
async def generate_speech(text, output_filename):
    communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
    await communicate.save(output_filename)

# --- STATIC WHITEBOARD FRAME GENERATOR ---
def create_whiteboard_slide(text, module_num, total_modules, width=960, height=540):
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

    # Text Content
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

# --- DIRECT FFMPEG ENGINE ---
def render_single_module_video(image_path, audio_path, output_mp4):
    """
    Executes raw FFmpeg directly with optimized flags:
    -tune stillimage: compresses static slides instantaneously.
    -c:a copy: transfers original AAC/MP3 audio without re-encoding.
    """
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-i", audio_path,
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_mp4
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def concatenate_videos_fast(video_paths, output_mp4):
    """
    Merges all module videos in milliseconds without re-encoding streams.
    """
    list_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt')
    for path in video_paths:
        list_file.write(f"file '{path}'\n")
    list_file.close()

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file.name,
        "-c", "copy",
        output_mp4
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    os.remove(list_file.name)

# --- WORKFLOW PIPELINE FOR SINGLE MODULE ---
def process_module(idx, chunk, total_chunks):
    mod_num = idx + 1

    # 1. Generate Script
    script = generate_module_explanation(chunk, mod_num, total_chunks)

    # 2. TTS Audio
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    asyncio.run(generate_speech(script, audio_path))

    # 3. Create Slide Image
    img_path = create_whiteboard_slide(script, mod_num, total_chunks)

    # 4. Render Video Segment via Direct FFmpeg
    module_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    render_single_module_video(img_path, audio_path, module_mp4)

    # Clean up intermediate files
    for f in [img_path, audio_path]:
        if os.path.exists(f):
            os.remove(f)

    return module_mp4

# --- STREAMLIT UI ---
st.title("⚡ Ultra-Fast AI Whiteboard Lecture Generator")

uploaded_file = st.file_uploader("Upload Document (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    if st.button("Generate Complete Lecture"):
        with st.spinner("Extracting text content..."):
            if uploaded_file.name.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                extracted_text = extract_text_from_docx(uploaded_file)

        if not extracted_text.strip():
            st.error("No text could be extracted.")
            st.stop()

        chunks = chunk_text_by_paragraphs(extracted_text, max_chars_per_chunk=4000)
        total_chunks = len(chunks)
        st.success(f"Document chunked into {total_chunks} modules.")

        module_mp4s = []
        prog = st.progress(0)
        status_text = st.empty()

        # Concurrent processing of modules (Script + TTS + FFmpeg Render)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_idx = {
                executor.submit(process_module, idx, chunk, total_chunks): idx 
                for idx, chunk in enumerate(chunks)
            }
            
            completed_count = 0
            results = [None] * total_chunks

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    mp4_path = future.result()
                    results[idx] = mp4_path
                    completed_count += 1
                    prog.progress(completed_count / total_chunks)
                    status_text.text(f"Completed Module {completed_count}/{total_chunks}...")
                except Exception as e:
                    st.error(f"Error processing module {idx+1}: {e}")

        # Filter out failed tasks
        module_mp4s = [r for r in results if r is not None]

        if module_mp4s:
            status_text.text("Merging video streams into final lecture...")
            final_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            
            # Instant FFmpeg Concatenation
            concatenate_videos_fast(module_mp4s, final_output)

            st.success("Lecture video generated successfully!")
            st.video(final_output)

            # Cleanup temporary MP4 clips
            for f in module_mp4s:
                if os.path.exists(f):
                    os.remove(f)

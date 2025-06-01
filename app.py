# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from streamlit_drawable_canvas import st_canvas
import uuid
import io
from fpdf import FPDF

# Set page config
st.set_page_config(page_title="AI Notepad", layout="centered")

st.title("✍️ AI Notepad: Handwriting to Text")
st.markdown("Draw handwritten text below and let AI convert it to clean typed text!")

# Load TrOCR model and processor
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return processor, model

processor, model = load_model()

# Initialize session state
if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = str(uuid.uuid4())
if "predicted_text" not in st.session_state:
    st.session_state["predicted_text"] = ""

# Drawing canvas
canvas_result = st_canvas(
    fill_color="black", 
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=600,
    drawing_mode="freedraw",
    key=st.session_state["canvas_key"],
)

# Image utilities
def crop_to_content(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    bbox = gray.getbbox()
    if bbox:
        return image.crop(bbox)
    return image

def get_image_from_canvas(canvas_data):
    if canvas_data is None:
        return None
    img_array = canvas_data[:, :, 0:3].astype(np.uint8)
    return Image.fromarray(img_array)

# TrOCR prediction
def predict_text(image: Image.Image) -> str:
    image = crop_to_content(image)
    image = ImageOps.invert(image.convert("L")).convert("RGB").resize((384, 384))
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    if st.button("🧠 Predict Text"):
        if canvas_result.image_data is not None:
            image = get_image_from_canvas(canvas_result.image_data)
            processed_image = ImageOps.invert(image.convert("L")).convert("RGB").resize((384, 384))
            # st.image(processed_image, caption="Preprocessed Image", use_column_width=True)
            text = predict_text(image)
            st.session_state["predicted_text"] += text + " "
            st.success(f"📝 Added: `{text}`")
        else:
            st.warning("Please draw something before predicting!")

    if st.button("Clear Canvas"):
        st.session_state["canvas_key"] = str(uuid.uuid4())
        st.rerun()

with col2:
    st.subheader("📝 Text Editor")
    edited_text = st.text_area("Predicted Text (Editable)", value=st.session_state["predicted_text"], height=300)
    st.session_state["predicted_text"] = edited_text

    colA, colB = st.columns(2)

    with colA:
        st.download_button(
            label="📥 Download as TXT",
            data=edited_text,
            file_name="handwritten_notes.txt"
        )

    with colB:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in edited_text.splitlines():
            pdf.multi_cell(0, 10, txt=line)

        pdf_bytes = pdf.output(dest="S").encode("latin1")  # Get PDF as bytes
        st.download_button(
            label="📄 Download as PDF",
            data=pdf_bytes,
            file_name="handwritten_notes.pdf",
            mime="application/pdf"
        )
    if st.button("🧹 Clear Text Editor"):
        st.session_state["predicted_text"] = ""
        st.rerun()

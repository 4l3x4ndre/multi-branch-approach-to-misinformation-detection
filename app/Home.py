import streamlit as st
import os
from PIL import Image
import torch
import pandas as pd

from app.utils import load_model, get_graphs_from_text
from corpus_truth_manipulation.config import CONFIG

st.set_page_config(page_title="Multimodal Misinformation Detector", page_icon="🕵️")

st.title("🕵️ Multimodal Misinformation Detector")
st.markdown("Upload an image and provide a text claim to analyze.")

# 1. Model Selection
# We use a specific model from Hugging Face
selected_model_filename = "global_trained-MMFakeBenchTest.pt"

# 2. Inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Image Input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_container_width=True)

with col2:
    st.subheader("Text Input")
    text_input = st.text_area("Enter the claim/caption text here...", height=200)

# 3. Analysis
if st.button("Run Analysis", type="primary"):
    if uploaded_file is None or not text_input.strip():
        st.warning("Please provide both an image and text.")
    else:
        with st.spinner('Loading model and processing...'):
            # Load Model
            model = load_model(selected_model_filename)
            if model is None:
                st.error("Failed to load model.")
                st.stop()
            
            # Process Graph
            st.info("Constructing Claim and Evidence Graphs (DBpedia)...")
            try:
                claim_batch, evidence_batch = get_graphs_from_text(text_input)
                if claim_batch is None:
                     st.error("Could not extract claim graph from text.")
                     st.stop()
            except Exception as e:
                st.error(f"Error during graph extraction: {e}")
                st.stop()

            # Prediction
            st.info("Running Model Inference...")
            try:
                # Prepare inputs
                with torch.no_grad():
                     outputs = model([image], [text_input], claim_batch, evidence_batch)
                     # outputs shape: [1, 4]
                     probs = outputs[0].tolist()
            except Exception as e:
                st.error(f"Error during inference: {e}")
                st.stop()

        # 4. Results
        st.success("Analysis Complete!")
        st.divider()
        
        # Output Neurons: 
        # 0. Image Realness (1=Real, 0=Fake)
        # 1. Claim Truthfulness (1=True, 0=False)
        # 2. Mismatch (1=Mismatch, 0=Matching/No Mismatch)
        # 3. Overall Truth (1=True/Original, 0=Fake)
        
        img_real_prob = probs[0]
        claim_real_prob = probs[1]
        mismatch_prob = probs[2]
        overall_truth_prob = probs[3]

        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        
        def get_color(prob, inverse=False):
            # If high prob is GOOD (Green), low is BAD (Red)
            # inverse: High prob is BAD (Red) (e.g. Mismatch)
            
            if inverse:
                # High prob = Red, Low prob = Green
                if prob > 0.5:
                    return f"background-color: rgba(255, 0, 0, {prob*0.5});"
                else:
                    return f"background-color: rgba(0, 255, 0, {(1-prob)*0.5});"
            else:
                # High prob = Green, Low prob = Red
                if prob > 0.5:
                     return f"background-color: rgba(0, 255, 0, {prob*0.5});"
                else:
                     return f"background-color: rgba(255, 0, 0, {(1-prob)*0.5});"

        with col_res1:
            st.markdown(f'<div style="padding:10px; border-radius:5px; {get_color(img_real_prob, inverse=False)}">'
                        f'<h4>Image Realness</h4>'
                        f'<h2>{img_real_prob:.2%}</h2>'
                        f'<p>{"Real" if img_real_prob > 0.5 else "Fake"}</p>'
                        '</div>', unsafe_allow_html=True)
            
        with col_res2:
            st.markdown(f'<div style="padding:10px; border-radius:5px; {get_color(claim_real_prob, inverse=False)}">'
                        f'<h4>Claim Truth</h4>'
                        f'<h2>{claim_real_prob:.2%}</h2>'
                        f'<p>{"True" if claim_real_prob > 0.5 else "False"}</p>'
                        '</div>', unsafe_allow_html=True)

        with col_res3:
            st.markdown(f'<div style="padding:10px; border-radius:5px; {get_color(mismatch_prob, inverse=True)}">'
                        f'<h4>Mismatch</h4>'
                        f'<h2>{mismatch_prob:.2%}</h2>'
                        f'<p>{"Mismatch" if mismatch_prob > 0.5 else "Matching"}</p>'
                        '</div>', unsafe_allow_html=True)

        with col_res4:
             st.markdown(f'<div style="padding:10px; border-radius:5px; {get_color(overall_truth_prob, inverse=False)}">'
                        f'<h4>Overall Truth</h4>'
                        f'<h2>{overall_truth_prob:.2%}</h2>'
                        f'<p>{"True" if overall_truth_prob > 0.5 else "Fake"}</p>'
                        '</div>', unsafe_allow_html=True)

        # Expanders for details
        with st.expander("See Extracted Claim Graph"):
             # We can visualize using extracting and converting to string or just showing nodes
             st.write(f"Nodes: {claim_batch.num_nodes}")
             st.write(f"Edges: {claim_batch.num_edges}")
             st.write(f"Text nodes: {claim_batch.node_texts}")
        
        with st.expander("See Extracted Evidence Graph"):
             st.write(f"Nodes: {evidence_batch.num_nodes}")
             st.write(f"Edges: {evidence_batch.num_edges}")
import streamlit as st

st.set_page_config(page_title="How To Guide", page_icon="📚")

st.title("📚 How To Guide & Project Overview")

st.markdown("""
### Disentangling Visual, Semantic, and Factual Deception
This project presents a novel **multi-branch multimodal architecture** for detecting misinformation. Unlike traditional methods that treat misinformation as a binary classification problem, this system explicitly models and disentangles three distinct aspects of deception:

1.  **Visual Manipulation** (Deepfake Detection)
2.  **Semantic Inconsistency** (Text-Image Mismatch)
3.  **Factual Falsity** (Claim Veracity against Knowledge Graphs)

---

### The Architecture

The model consists of three specialized branches that operate in parallel:

#### 1. Deepfake Detection Branch
*   **Goal:** Assess the visual authenticity of the image.
*   **Method:** Uses a state-of-the-art vision transformer (SigLIP) fine-tuned on deepfake datasets.
*   **Output:** Probability that the image is real vs. manipulated/generated.

#### 2. Text-Image Consistency Branch
*   **Goal:** Detect "out-of-context" misinformation where a real image is paired with a misleading caption.
*   **Method:** Leverages **CLIP (Contrastive Language-Image Pre-training)** to measure the semantic alignment between the visual content and the textual claim.
*   **Output:** Probability of a mismatch between text and image.

#### 3. Encyclopedic (Claim Veracity) Branch
*   **Goal:** Ground textual claims in verifiable external knowledge.
*   **Method:** 
    *   Constructs a **Claim Graph** from the input text using linguistic analysis (spaCy).
    *   Constructs an **Evidence Graph** by dynamically querying **DBpedia** for entities mentioned in the text.
    *   Uses a **Graph Neural Network (GNN)** with cross-attention to reason about the consistency between the claim and the evidence.
*   **Output:** Probability that the claim is factually true.

#### Final Classification
The outputs from these three branches are fused to predict the **Overall Truthfulness** of the multimedia content.

---

### How to Use the Demo

1.  **Upload an Image:** Provide the visual component of the news item or post.
2.  **Enter Text:** Type or paste the accompanying caption or claim.
3.  **Run Analysis:** The system will process the inputs and display:
    *   **Image Realness:** Is the image itself authentic?
    *   **Claim Truth:** Is the text supported by factual knowledge?
    *   **Mismatch:** Does the text contradict or misrepresent the image?
    *   **Overall Truth:** Is the content as a whole trustworthy?

### Interpretation of Results
*   **Green Background:** Indicates a "Good" or "Authentic" signal (Real Image, True Claim, Matching Content, Overall True).
*   **Red Background:** Indicates a "Bad" or "Fake" signal (Deepfake, False Claim, Mismatch, Overall Fake).

---

### References
*   EGMMG pipeline: **Duwal, S., Shopnil, M. N. S., Tyagi, A., & Proma, A. M. (2025). *Evidence-Grounded Multimodal Misinformation Detection with Attention-Based GNNs*. arXiv. [https://doi.org/10.48550/arXiv.2505.18221](https://doi.org/10.48550/arXiv.2505.18221)**
*   DeepFakeDetector on [HuggingFace](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1)
*   COSMOS: Aneja, S., Bregler, C., & Nießner, M. (2021). *COSMOS: Catching Out-of-Context Misinformation with Self-Supervised Learning*. arXiv. [https://doi.org/10.48550/ARXIV.2101.06278](https://doi.org/10.48550/ARXIV.2101.06278)
*   MMFakeBench: **Liu, X., Li, Z., Li, P., Huang, H., Xia, S., Cui, X., Huang, L., Deng, W., & He, Z. (2024). *MMFakeBench: A Mixed-Source Multimodal Misinformation Detection Benchmark for LVLMs*. arXiv. [https://doi.org/10.48550/arXiv.2406.08772](https://doi.org/10.48550/arXiv.2406.08772)**

*   Knowledge Base: [DBpedia](https://www.dbpedia.org/).
*   OpenAI-CLIP:  **Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. arXiv. [https://doi.org/10.48550/arXiv.2103.00020](https://doi.org/10.48550/arXiv.2103.00020)**

""")

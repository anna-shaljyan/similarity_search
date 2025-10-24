import streamlit as st
import os, numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# --- Simple password protection ---
st.title("🔐 Educational Design Similarity Tool")
password = st.text_input("Enter password:", type="password")
if password != "jdsprinting123456":   # example
    st.warning("Incorrect password.")
    st.stop()


# --- Setup ---
os.makedirs("images_db", exist_ok=True)
model = SentenceTransformer("clip-ViT-B-32")

# --- Load existing embeddings ---
if os.path.exists("embeddings.npy") and os.path.exists("filenames.npy"):
    embeddings = np.load("embeddings.npy")
    file_names = np.load("filenames.npy")
else:
    embeddings, file_names = np.array([]), np.array([])

# --- Upload new image ---
uploaded_file = st.file_uploader("📤 Upload a new placemat image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB").resize((512, 512))
    st.image(query_img, caption="Uploaded image", use_column_width=True)
    query_emb = model.encode(query_img, convert_to_tensor=False, normalize_embeddings=True)

    # If we have DB images, search for similarity
    if embeddings.size > 0:
        similarities = cosine_similarity([query_emb], embeddings)[0]
        top_idx = np.argsort(similarities)[::-1][:5]

        st.write("### 🔎 Top 5 Similar Designs:")
        cols = st.columns(5)
        for i, idx in enumerate(top_idx):
            img_path = os.path.join("images_db", file_names[idx])
            cols[i].image(Image.open(img_path), caption=f"Sim: {similarities[idx]:.2f}")
    else:
        st.warning("No existing images in database yet.")

    # Option to add image to DB
    if st.button("✅ Add this image to database"):
        file_path = os.path.join("images_db", uploaded_file.name)
        query_img.save(file_path)
        new_emb = np.expand_dims(query_emb, axis=0)

        if embeddings.size == 0:
            embeddings = new_emb
            file_names = np.array([uploaded_file.name])
        else:
            embeddings = np.vstack([embeddings, new_emb])
            file_names = np.append(file_names, uploaded_file.name)

        np.save("embeddings.npy", embeddings)
        np.save("filenames.npy", file_names)
        st.success("Image added successfully!")

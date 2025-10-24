import streamlit as st
import os, numpy as np, io, zipfile, requests, json
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Educational Design Similarity Tool", layout="wide")

# =========================================================
# 🧠 PASSWORD SETUP (first-time create → stored securely in .password file)
# =========================================================
st.title("🔐 Educational Design Similarity Tool")

pw_file = ".password"

if not os.path.exists(pw_file):
    st.info("🪄 No password set yet. Create one to secure your app.")
    new_pw = st.text_input("Create a new password:", type="password")
    confirm_pw = st.text_input("Confirm password:", type="password")
    if st.button("Save password"):
        if new_pw and new_pw == confirm_pw:
            with open(pw_file, "w") as f:
                f.write(new_pw.strip())
            st.success("✅ Password created successfully! Please reload the app.")
            st.stop()
        else:
            st.error("❌ Passwords do not match.")
            st.stop()
else:
    saved_pw = open(pw_file).read().strip()
    password = st.text_input("Enter password:", type="password")
    if password != saved_pw:
        st.warning("Incorrect password.")
        st.stop()

# =========================================================
# 📦 SETUP & LOAD DATABASE
# =========================================================
image_dir = "images_db"

# Auto-download images_db from GitHub if not found
if not os.path.exists(image_dir):
    st.info("📦 Downloading images_db from GitHub...")
    try:
        url = "https://github.com/anna-shaljyan/similarity_search/archive/refs/heads/main.zip"
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(".")
        os.rename("similarity_search-main/images_db", image_dir)
        st.success("✅ images_db folder successfully downloaded!")
    except Exception as e:
        st.error(f"❌ Failed to load images_db: {e}")
        st.stop()

# List image files
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if not image_files:
    st.error("❌ No images found in 'images_db' folder.")
    st.stop()
else:
    st.success(f"✅ Found {len(image_files)} images in the database.")

# =========================================================
# 🧠 MODEL & EMBEDDINGS
# =========================================================
model = SentenceTransformer("clip-ViT-B-32")

if os.path.exists("embeddings.npy") and os.path.exists("filenames.npy"):
    embeddings = np.load("embeddings.npy")
    file_names = np.load("filenames.npy")
    st.info(f"Loaded existing embeddings for {len(file_names)} images.")
else:
    st.info("⚙️ Creating embeddings for database images (first run)...")
    imgs = []
    for f in image_files:
        try:
            img_path = os.path.join(image_dir, f)
            img = Image.open(img_path).convert("RGB").resize((512, 512))
            imgs.append(img)
        except Exception:
            st.warning(f"Skipping unreadable image: {f}")
    embeddings = model.encode(imgs, convert_to_numpy=True, normalize_embeddings=True)
    file_names = np.array(image_files)
    np.save("embeddings.npy", embeddings)
    np.save("filenames.npy", file_names)
    st.success("✅ Embeddings created and saved!")

# =========================================================
# 📤 UPLOAD & SEARCH
# =========================================================
uploaded_file = st.file_uploader("📤 Upload a new image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB").resize((512, 512))
    st.image(query_img, caption="Uploaded image", use_column_width=True)
    query_emb = model.encode(query_img, convert_to_tensor=False, normalize_embeddings=True)

    # --- Find top 5 similar designs ---
    if embeddings.size > 0:
        similarities = cosine_similarity([query_emb], embeddings)[0]
        top_idx = np.argsort(similarities)[::-1][:5]

        st.markdown("### 🔎 Top 5 Similar Designs:")
        cols = st.columns(5)
        for i, idx in enumerate(top_idx):
            img_path = os.path.join(image_dir, file_names[idx])
            cols[i].image(Image.open(img_path), caption=f"Sim: {similarities[idx]:.2f}")
    else:
        st.warning("No existing images in database yet.")

    # --- Add to database ---
    if st.checkbox("✅ Add this image to database"):
        file_path = os.path.join(image_dir, uploaded_file.name)
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
        st.success("🧩 Image added to local database!")

        # Optional: show count
        st.info(f"Total images in database: {len(file_names)}")

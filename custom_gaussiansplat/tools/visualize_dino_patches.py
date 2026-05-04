import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from transformers import AutoImageProcessor, AutoModel

st.set_page_config(layout="wide", page_title="DINOv2 Feature Explorer")

if "selected_points" not in st.session_state:
    st.session_state.selected_points = []
if "last_click_signature" not in st.session_state:
    st.session_state.last_click_signature = None
if "selection_context" not in st.session_state:
    st.session_state.selection_context = None

# ---------------------------------------------------------------------------
# 1. Core Mechanics & Caching
# ---------------------------------------------------------------------------
# st.cache_resource keeps the model in VRAM/RAM across user interactions
@st.cache_resource
def load_dino_model(model_name="facebook/dinov2-base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    patch_size = model.config.patch_size if hasattr(model.config, "patch_size") else 14
    return processor, model, device, patch_size

# st.cache_data ensures we only run the heavy forward pass ONCE per image upload
@st.cache_data
def extract_features(_image, _processor, _model, device, patch_size):
    orig_w, orig_h = _image.size
    
    # Force aspect ratio to be a multiple of patch size
    new_h = (orig_h // patch_size) * patch_size
    new_w = (orig_w // patch_size) * patch_size
    new_h, new_w = max(new_h, patch_size), max(new_w, patch_size)
    
    img_resized = _image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    inputs = _processor(images=img_resized, return_tensors="pt", do_resize=False, do_center_crop=False).to(device)
    
    with torch.no_grad():
        outputs = _model(**inputs)
        hidden_states = outputs.last_hidden_state
        
    h_feat, w_feat = new_h // patch_size, new_w // patch_size
    expected_tokens = h_feat * w_feat
    token_count = hidden_states.shape[1]
    
    # Strictly isolate spatial patches, discarding the CLS token
    if token_count == expected_tokens + 1:
        spatial_patches = hidden_states[0, 1:, :]
    else:
        spatial_patches = hidden_states[0, :, :]
        
    # Reshape into a 2D spatial grid: [H_feat, W_feat, D]
    patch_grid = spatial_patches.reshape(h_feat, w_feat, -1)
    
    return patch_grid, img_resized, h_feat, w_feat

# ---------------------------------------------------------------------------
# 2. UI Layout & State Management
# ---------------------------------------------------------------------------
st.title("DINOv2 Interactive Feature Explorer")
st.markdown("Upload an image, click anywhere to select a prototype feature, and adjust the threshold to see single-shot segmentation.")

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Model", ["facebook/dinov2-small", "facebook/dinov2-base", "facebook/dinov2-large"], index=1)
    threshold = st.slider("Cosine Similarity Threshold", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    clear_selection = st.button("Clear Selected Points", use_container_width=True)

processor, model, device, patch_size = load_dino_model(model_choice)

if uploaded_file is not None:
    # Process image
    pil_image = Image.open(uploaded_file).convert("RGB")
    patch_grid, img_resized, h_feat, w_feat = extract_features(pil_image, processor, model, device, patch_size)

    selection_context = f"{uploaded_file.name}:{uploaded_file.size}:{model_choice}:{img_resized.width}x{img_resized.height}"
    if st.session_state.selection_context != selection_context:
        st.session_state.selection_context = selection_context
        st.session_state.selected_points = []
        st.session_state.last_click_signature = None

    if clear_selection:
        st.session_state.selected_points = []
        st.session_state.last_click_signature = None
        st.rerun()

    # Keep all panels visually consistent in a 3-column layout.
    click_display_width = min(img_resized.width, 520)
    click_display_height = max(1, int(round(img_resized.height * click_display_width / img_resized.width)))
    plot_height = 6 * (img_resized.height / img_resized.width)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Click to Select Prototypes")
        # This component returns the (x, y) coordinates of the user's click
        value = streamlit_image_coordinates(img_resized, key="pil", width=click_display_width)

        if st.session_state.selected_points:
            st.caption(f"Selected points: {len(st.session_state.selected_points)}")

    if value is not None:
        click_signature = (selection_context, int(value["x"]), int(value["y"]))
        if click_signature != st.session_state.last_click_signature:
            # Coordinates returned by the click widget are in displayed-image space.
            # Rescale them back to the resized image used for feature extraction.
            click_x = int(np.clip(round(value['x'] * (img_resized.width / click_display_width)), 0, img_resized.width - 1))
            click_y = int(np.clip(round(value['y'] * (img_resized.height / click_display_height)), 0, img_resized.height - 1))

            # Map pixel coordinates to patch grid indices
            patch_x = min(click_x // patch_size, w_feat - 1)
            patch_y = min(click_y // patch_size, h_feat - 1)

            st.session_state.selected_points.append(
                {
                    "click_x": click_x,
                    "click_y": click_y,
                    "patch_x": int(patch_x),
                    "patch_y": int(patch_y),
                }
            )
            st.session_state.last_click_signature = click_signature
    
    selected_points = st.session_state.selected_points

    if selected_points:
        # -------------------------------------------------------------------
        # 3. Mathematical Logic: multi-point prototypes to patch similarities
        # -------------------------------------------------------------------
        prototype_features = [patch_grid[p["patch_y"], p["patch_x"], :] for p in selected_points]
        prototype_matrix = torch.stack(prototype_features, dim=0)                # [K, D]

        # Normalize vectors for cosine similarity
        prototype_norm = F.normalize(prototype_matrix, p=2, dim=1)               # [K, D]
        grid_flat = patch_grid.reshape(-1, patch_grid.shape[-1])                 # [H*W, D]
        grid_norm = F.normalize(grid_flat, p=2, dim=1)                           # [H*W, D]

        # Compare each patch against all selected prototypes, then keep best match.
        similarity_matrix = torch.mm(grid_norm, prototype_norm.T)                # [H*W, K]
        similarities = similarity_matrix.max(dim=1).values                       # [H*W]
        sim_grid = similarities.reshape(h_feat, w_feat).cpu().numpy()            # [H, W]
        
        # -------------------------------------------------------------------
        # 4. Rendering the Output
        # -------------------------------------------------------------------
        with col2:
            st.subheader("2. Similarity Heatmap")
            fig, ax = plt.subplots(figsize=(6, plot_height))
            ax.imshow(img_resized)
            # Overlay the raw similarity values as a heatmap
            ax.imshow(sim_grid, cmap='jet', alpha=0.5, extent=(0, img_resized.width, img_resized.height, 0))

            for idx, point in enumerate(selected_points):
                ax.plot(
                    point["click_x"],
                    point["click_y"],
                    'ro',
                    markersize=7,
                    markeredgecolor='white',
                    markeredgewidth=1.5,
                )
                ax.text(point["click_x"] + 4, point["click_y"] - 4, str(idx + 1), color='white', fontsize=9, weight='bold')

            ax.axis('off')
            st.pyplot(fig, use_container_width=True)
            
        with col3:
            st.subheader(f"3. Hard Mask (> {threshold})")
            # Create binary mask based on the sidebar slider
            hard_mask = (sim_grid > threshold).astype(float)
            
            # Upsample mask to original image size for display
            hard_mask_tensor = torch.tensor(hard_mask).unsqueeze(0).unsqueeze(0)
            upsampled_mask = F.interpolate(hard_mask_tensor, size=(img_resized.height, img_resized.width), mode='nearest').squeeze().numpy()
            
            fig2, ax2 = plt.subplots(figsize=(6, plot_height))
            ax2.imshow(img_resized)
            # Black out anything below the threshold
            ax2.imshow(upsampled_mask, cmap='gray', alpha=0.7, extent=(0, img_resized.width, img_resized.height, 0), vmin=0, vmax=1)
            ax2.axis('off')
            st.pyplot(fig2, use_container_width=True)
            
        # Brutally honest data dump for debugging
        st.write("---")
        latest_point = selected_points[-1]
        st.write(f"**Selected Points:** {len(selected_points)}")
        st.write(f"**Latest Click Coordinates:** X: {latest_point['click_x']}, Y: {latest_point['click_y']}")
        st.write(f"**Latest Mapped Patch Index:** X: {latest_point['patch_x']}, Y: {latest_point['patch_y']}")
        st.write(f"**Feature Dimension:** {patch_grid.shape[-1]}")
    else:
        with col2:
            st.info("Select one or more points to render similarity heatmap.")
        with col3:
            st.info("Select one or more points to render hard mask.")
else:
    st.info("Upload an image to begin.")

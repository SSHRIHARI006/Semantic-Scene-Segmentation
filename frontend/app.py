import streamlit as st
import cv2
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_inference import predict_segmentation
from path_planner import compute_path, AStarPathPlanner
from visualization import (overlay_segmentation, draw_path_on_image, 
                          visualize_obstacle_grid, create_comparison_grid)

# Page configuration
st.set_page_config(
    page_title="Autonomous Terrain Navigation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🚗 Off-Road Autonomous Terrain Navigation System")
st.markdown("""
This system performs semantic segmentation on terrain images and computes optimal navigation paths 
avoiding obstacles like rocks and trees.
""")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Path planning settings
    st.subheader("Path Planning")
    overlay_alpha = st.slider("Segmentation Overlay Transparency", 0.0, 1.0, 0.5, 0.05)
    
    start_y = st.slider("Start Position (from bottom)", 10, 100, 20)
    goal_y = st.slider("Goal Position (from top)", 10, 100, 20)
    
    st.markdown("---")
    st.markdown("""
    ### Class Legend
    - 🟤 **Brown/Tan**: Navigable terrain
    - 🟢 **Green**: Vegetation (obstacle)
    - ⚫ **Gray**: Rocks (obstacle)
    - 🔵 **Sky Blue**: Sky
    """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Upload off-road terrain images for best results")

# File uploader
uploaded_file = st.file_uploader(
    "📁 Upload Terrain Image",
    type=["jpg", "png", "jpeg", "bmp"],
    help="Upload an image of off-road terrain"
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    if image is None:
        st.error("❌ Error loading image. Please try another file.")
        st.stop()
    
    height, width = image.shape[:2]
    
    # Show processing message
    with st.spinner("🔄 Processing image..."):
        start_time = time.time()
        
        # Step 1: Segmentation
        with st.spinner("🎨 Running semantic segmentation..."):
            seg_mask, colored_mask, obstacle_grid = predict_segmentation(image)
        
        # Step 2: Create overlay
        overlay = overlay_segmentation(image, colored_mask, alpha=overlay_alpha)
        
        # Step 3: Path planning
        with st.spinner("🗺️ Computing optimal path..."):
            start_pos = (width // 2, height - start_y)
            goal_pos = (width // 2, goal_y)
            path_image, path = compute_path(image, obstacle_grid, start_pos, goal_pos)
        
        processing_time = time.time() - start_time
    
    # Success message
    st.success(f"✅ Processing completed in {processing_time:.2f} seconds!")
    
    # Display results in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, channels="BGR", use_container_width=True)
        st.caption(f"Resolution: {width}x{height}")
    
    with col2:
        st.subheader("🎨 Segmentation Map")
        st.image(overlay, channels="BGR", use_container_width=True)
        
        # Count navigable pixels
        navigable_pixels = np.sum(obstacle_grid == 0)
        total_pixels = obstacle_grid.size
        navigable_percent = (navigable_pixels / total_pixels) * 100
        st.caption(f"Navigable Area: {navigable_percent:.1f}%")
    
    with col3:
        st.subheader("🗺️ Optimal Path")
        st.image(path_image, channels="BGR", use_container_width=True)
        
        if path is not None:
            path_length = len(path)
            st.caption(f"Path Length: {path_length} pixels")
        else:
            st.caption("⚠️ No valid path found")
    
    # Additional information section
    st.markdown("---")
    
    # Expandable sections for detailed info
    with st.expander("📊 Detailed Analysis"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Terrain Classification")
            # Count pixels per class
            unique, counts = np.unique(seg_mask, return_counts=True)
            class_names = ["Unknown", "Forest", "Vegetation", "Sand", "Terrain", 
                          "Olive", "Rock-Brown", "Rock-Gray", "Sienna", "Sky"]
            
            for cls, count in zip(unique, counts):
                if cls < len(class_names):
                    percentage = (count / seg_mask.size) * 100
                    st.write(f"**{class_names[cls]}**: {percentage:.2f}%")
        
        with col_b:
            st.subheader("Navigation Metrics")
            st.write(f"**Image Size**: {width} x {height} pixels")
            st.write(f"**Total Pixels**: {total_pixels:,}")
            st.write(f"**Navigable Pixels**: {navigable_pixels:,}")
            st.write(f"**Obstacle Pixels**: {total_pixels - navigable_pixels:,}")
            
            if path is not None:
                st.write(f"**Path Found**: ✅ Yes")
                st.write(f"**Path Length**: {len(path)} pixels")
                # Calculate path safety (how far from obstacles on average)
                st.write(f"**Start**: ({start_pos[0]}, {start_pos[1]})")
                st.write(f"**Goal**: ({goal_pos[0]}, {goal_pos[1]})")
            else:
                st.write(f"**Path Found**: ❌ No")
                st.write("*Try adjusting start/goal positions*")
    
    with st.expander("💾 Download Results"):
        st.write("Download processed images:")
        
        # Convert images for download
        _, seg_encoded = cv2.imencode('.png', overlay)
        _, path_encoded = cv2.imencode('.png', path_image)
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.download_button(
                label="⬇️ Download Segmentation",
                data=seg_encoded.tobytes(),
                file_name="segmentation_overlay.png",
                mime="image/png"
            )
        
        with col_d2:
            st.download_button(
                label="⬇️ Download Path Visualization",
                data=path_encoded.tobytes(),
                file_name="navigation_path.png",
                mime="image/png"
            )

else:
    # Show sample usage
    st.info("👆 Please upload an image to begin")
    
    st.markdown("""
    ### How it works:
    
    1. **📸 Image Upload**: Upload a terrain/off-road image
    2. **🎨 Semantic Segmentation**: AI model classifies each pixel (terrain, rocks, trees, etc.)
    3. **🗺️ Obstacle Mapping**: Converts segmentation into navigable vs obstacle grid
    4. **🧭 Path Planning**: A* algorithm finds optimal safe route
    5. **✨ Visualization**: Displays results with path overlay
    
    ### Features:
    - ✅ Real-time semantic segmentation using DINOv2 + custom decoder
    - ✅ A* pathfinding algorithm for optimal navigation
    - ✅ Interactive parameter adjustment
    - ✅ Detailed terrain analysis
    - ✅ Export results
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit, PyTorch, and DINOv2</p>
</div>
""", unsafe_allow_html=True)
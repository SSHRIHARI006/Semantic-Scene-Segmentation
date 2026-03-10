import cv2
import numpy as np


def overlay_segmentation(image, segmentation, alpha=0.5):
    """
    Overlay segmentation map on original image
    
    Args:
        image: Original BGR image
        segmentation: RGB segmentation mask
        alpha: Transparency factor (0-1). Higher = more segmentation visible
        
    Returns:
        Blended overlay image
    """
    # Ensure same dimensions
    if image.shape[:2] != segmentation.shape[:2]:
        segmentation = cv2.resize(segmentation, (image.shape[1], image.shape[0]))
    
    # Blend images
    overlay = cv2.addWeighted(image, 1 - alpha, segmentation, alpha, 0)
    
    return overlay


def create_side_by_side(images, labels=None, border_width=2):
    """
    Create side-by-side visualization of multiple images
    
    Args:
        images: List of images (BGR format)
        labels: Optional list of labels for each image
        border_width: Width of border between images
        
    Returns:
        Combined image
    """
    if not images:
        return None
    
    # Ensure all images have same height
    max_height = max(img.shape[0] for img in images)
    resized_images = []
    
    for img in images:
        if img.shape[0] != max_height:
            aspect = img.shape[1] / img.shape[0]
            new_width = int(max_height * aspect)
            img = cv2.resize(img, (new_width, max_height))
        resized_images.append(img)
    
    # Add labels if provided
    if labels:
        labeled_images = []
        for img, label in zip(resized_images, labels):
            img_copy = img.copy()
            cv2.putText(img_copy, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            labeled_images.append(img_copy)
        resized_images = labeled_images
    
    # Add borders
    if border_width > 0:
        bordered_images = []
        for img in resized_images:
            bordered = cv2.copyMakeBorder(
                img, 0, 0, border_width, 0,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            bordered_images.append(bordered)
        resized_images = bordered_images[:-1] + [resized_images[-1]]  # Remove border from last image
    
    # Concatenate horizontally
    combined = np.hstack(resized_images)
    
    return combined


def visualize_obstacle_grid(obstacle_grid):
    """
    Visualize obstacle grid as binary image
    
    Args:
        obstacle_grid: Binary grid (1=obstacle, 0=navigable)
        
    Returns:
        RGB visualization
    """
    height, width = obstacle_grid.shape
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Navigable areas: Light gray
    vis[obstacle_grid == 0] = [200, 200, 200]
    
    # Obstacles: Dark red
    vis[obstacle_grid == 1] = [0, 0, 128]
    
    return vis


def draw_path_on_image(image, path, start_pos=None, goal_pos=None, 
                       path_color=(0, 255, 0), path_thickness=3):
    """
    Draw navigation path on image with start and goal markers
    
    Args:
        image: Original BGR image
        path: List of path coordinates [(row, col), ...]
        start_pos: Start position (col, row) for marker
        goal_pos: Goal position (col, row) for marker
        path_color: RGB color for path line
        path_thickness: Thickness of path line
        
    Returns:
        Image with path overlay
    """
    result = image.copy()
    
    if path is not None and len(path) > 1:
        # Convert path from (row, col) to (col, row) for drawing
        path_points = [(p[1], p[0]) for p in path]
        
        # Draw path line
        for i in range(len(path_points) - 1):
            cv2.line(result, path_points[i], path_points[i + 1], 
                    path_color, path_thickness)
        
        # Draw path as semi-transparent overlay
        overlay = result.copy()
        for point in path_points:
            cv2.circle(overlay, point, path_thickness, path_color, -1)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
    
    # Draw start marker
    if start_pos is not None:
        cv2.circle(result, start_pos, 12, (255, 0, 0), -1)  # Blue circle
        cv2.circle(result, start_pos, 12, (255, 255, 255), 2)  # White border
        cv2.putText(result, "START", (start_pos[0] - 35, start_pos[1] + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, "START", (start_pos[0] - 35, start_pos[1] + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
    
    # Draw goal marker
    if goal_pos is not None:
        cv2.circle(result, goal_pos, 12, (0, 0, 255), -1)  # Red circle
        cv2.circle(result, goal_pos, 12, (255, 255, 255), 2)  # White border
        cv2.putText(result, "GOAL", (goal_pos[0] - 30, goal_pos[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, "GOAL", (goal_pos[0] - 30, goal_pos[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    
    return result


def create_comparison_grid(images, titles=None, grid_cols=2):
    """
    Create a grid layout of multiple images for comparison
    
    Args:
        images: List of images (BGR format)
        titles: Optional list of titles for each image
        grid_cols: Number of columns in grid
        
    Returns:
        Combined grid image
    """
    if not images:
        return None
    
    n_images = len(images)
    grid_rows = (n_images + grid_cols - 1) // grid_cols
    
    # Find max dimensions
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    
    # Create grid
    grid = []
    for row in range(grid_rows):
        row_images = []
        for col in range(grid_cols):
            idx = row * grid_cols + col
            if idx < n_images:
                img = images[idx].copy()
                
                # Resize to match dimensions
                if img.shape[0] != max_height or img.shape[1] != max_width:
                    img = cv2.resize(img, (max_width, max_height))
                
                # Add title if provided
                if titles and idx < len(titles):
                    cv2.rectangle(img, (0, 0), (max_width, 40), (0, 0, 0), -1)
                    cv2.putText(img, titles[idx], (10, 28),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                row_images.append(img)
            else:
                # Fill with black if grid not complete
                row_images.append(np.zeros((max_height, max_width, 3), dtype=np.uint8))
        
        grid.append(np.hstack(row_images))
    
    return np.vstack(grid)


def add_legend(image, class_names, colors, position='right'):
    """
    Add color legend to image
    
    Args:
        image: Input BGR image
        class_names: List of class names
        colors: List of RGB colors corresponding to classes
        position: Position of legend ('right', 'bottom')
        
    Returns:
        Image with legend
    """
    img_copy = image.copy()
    height, width = img_copy.shape[:2]
    
    if position == 'right':
        # Create legend panel
        legend_width = 200
        legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 255
        
        # Add legend items
        y_offset = 50
        for i, (name, color) in enumerate(zip(class_names, colors)):
            if y_offset + 30 < height:
                # Draw color box
                cv2.rectangle(legend, (20, y_offset), (50, y_offset + 20), 
                            color.tolist(), -1)
                cv2.rectangle(legend, (20, y_offset), (50, y_offset + 20), 
                            (0, 0, 0), 1)
                
                # Draw text
                cv2.putText(legend, name, (60, y_offset + 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                y_offset += 30
        
        # Add title
        cv2.putText(legend, "Legend", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Concatenate
        result = np.hstack([img_copy, legend])
    
    else:  # bottom
        # Create legend panel
        legend_height = 100
        legend = np.ones((legend_height, width, 3), dtype=np.uint8) * 255
        
        # Add legend items horizontally
        x_offset = 20
        for name, color in zip(class_names, colors):
            if x_offset + 150 < width:
                # Draw color box
                cv2.rectangle(legend, (x_offset, 30), (x_offset + 30, 50), 
                            color.tolist(), -1)
                cv2.rectangle(legend, (x_offset, 30), (x_offset + 30, 50), 
                            (0, 0, 0), 1)
                
                # Draw text
                cv2.putText(legend, name, (x_offset + 35, 45),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                x_offset += 150
        
        # Concatenate
        result = np.vstack([img_copy, legend])
    
    return result


def add_legend(image, class_names, colors, position='right'):
    """
    Add color legend to image
    
    Args:
        image: Input BGR image
        class_names: List of class names
        colors: List of RGB colors
        position: 'right' or 'bottom'
        
    Returns:
        Image with legend
    """
    img = image.copy()
    height, width = img.shape[:2]
    
    box_size = 20
    text_offset = 30
    padding = 10
    
    if position == 'right':
        legend_width = 200
        legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 255
        
        y_start = 20
        for i, (name, color) in enumerate(zip(class_names, colors)):
            y = y_start + i * (box_size + padding)
            if y + box_size > height:
                break
            
            # Draw color box (convert RGB to BGR for cv2)
            cv2.rectangle(legend, (10, y), (10 + box_size, y + box_size),
                         (int(color[2]), int(color[1]), int(color[0])), -1)
            
            # Draw text
            cv2.putText(legend, name, (10 + text_offset, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Combine with image
        combined = np.hstack([img, legend])
    else:
        # Bottom legend
        legend_height = 100
        legend = np.ones((legend_height, width, 3), dtype=np.uint8) * 255
        
        x_start = 20
        for i, (name, color) in enumerate(zip(class_names, colors)):
            x = x_start + i * 150
            if x + box_size > width:
                break
            
            # Draw color box
            cv2.rectangle(legend, (x, 20), (x + box_size, 20 + box_size),
                         (int(color[2]), int(color[1]), int(color[0])), -1)
            
            # Draw text
            cv2.putText(legend, name, (x + text_offset, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Combine with image
        combined = np.vstack([img, legend])
    
    return combined

import cv2
import numpy as np
import os

def crop_center_width(input_path, output_path):
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        raise Exception(f"Could not load image from {input_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Calculate crop dimensions (60% of width from center)
    crop_width = int(width * 0.8)  # 60% of original width
    start_x = (width - crop_width) // 2  # Start point for center crop
    
    # Crop the image
    cropped_img = img[:, start_x:start_x+crop_width]
    
    # Save the cropped image
    cv2.imwrite(output_path, cropped_img)
    print(f"Original size: {width}x{height}")
    print(f"Cropped size: {crop_width}x{height}")
    print(f"Cropped image saved to: {output_path}")

if __name__ == "__main__":
    input_path = 'uploads/target_image.webp'
    output_path = 'results/cropped.png'
    
    try:
        crop_center_width(input_path, output_path)
    except Exception as e:
        print(f"Error: {str(e)}")
from flask import Flask, render_template, request, jsonify, redirect, url_for, make_response
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import uuid
import base64
import threading
import requests
from PIL import Image
import io
import firebase_admin
from firebase_admin import credentials, storage
import time
import logging
import datetime
import qrcode
from io import BytesIO
import shutil
from gfpgan import GFPGANer

app = Flask(__name__)

# Disable caching for specific routes
@app.after_request
def add_header(response):
    if request.path == '/static/target_image.png' or request.path == '/static/qr_code.png':
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# Initialize Firebase
cred = credentials.Certificate('./serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'ai-photo-7495c.appspot.com'
})
bucket = storage.bucket()

# Initialize face analysis and swapping models
app_insightface = FaceAnalysis(name='buffalo_l')
app_insightface.prepare(ctx_id=0, det_size=(640, 640))
model_path = "./inswapper_128.onnx"
swapper = insightface.model_zoo.get_model(model_path)

# Initialize GFPGAN for face enhancement
gfpganer = GFPGANer(model_path='GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2)

# Global variables for results
result_img_path = None
result_img_path_firebase = None
result_lock = threading.Lock()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_unique_filename():
    return str(uuid.uuid4()) + '.png'

def generate_target_image(custom_prompts=None):
    # "b54780508fd1d61abff1eb2eaa6eaa4b157ffb81e4328a4e7a428cb227cdd89053193f68473f838eb0466c2174195482"
    clipdrop_api_key = 'b54780508fd1d61abff1eb2eaa6eaa4b157ffb81e4328a4e7a428cb227cdd89053193f68473f838eb0466c2174195482'
    predefined_prompts_str = "photorealistic concept art, high quality digital art, cinematic, hyperrealism, photorealism, Nikon D850, 8K., sharp focus, emitting diodes, artillery, motherboard, by pascal blanche rutkowski repin artstation hyperrealism painting concept art of detailed character design matte painting, 4 k resolution"

    all_prompts = predefined_prompts_str
    if custom_prompts:
        all_prompts += "\n" + custom_prompts

    headers = {
        'x-api-key': clipdrop_api_key,
        'accept': 'image/webp',
        'x-clipdrop-width': '400',  # Desired width in pixels
        'x-clipdrop-height': '600',  # Desired height in pixels
       
    }

    data = {'prompt': (None, all_prompts, 'text/plain')}
    response = requests.post('https://clipdrop-api.co/text-to-image/v1', files=data, headers=headers)

    if response.ok:
        with open('static/target_image.webp', 'wb') as f:
            f.write(response.content)
    else:
        logger.error(f"Failed to generate target image: {response.status_code}")

def upload_image_to_firebase(image_path):
    unique_filename = generate_unique_filename()
    try:
        blob = bucket.blob('ig-ai-images/' + unique_filename)
        blob.upload_from_filename(image_path)
        
        # Save a local copy
        print_img_path_local = os.path.join('static', 'print.png')
        shutil.copyfile(image_path, print_img_path_local)

        expires_in = datetime.timedelta(days=1)
        url = blob.generate_signed_url(expires_in)
        return url
    except Exception as e:
        logger.error(f"Failed to upload to Firebase: {str(e)}")
        return None

def enhance_face(image):
    logger.info("Starting face enhancement...")
    _, _, restored_img = gfpganer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
    
    if isinstance(restored_img, Image.Image):
        restored_img = np.array(restored_img)
    
    if isinstance(restored_img, np.ndarray):
        logger.info("Face enhancement completed.")
        return restored_img
    else:
        raise ValueError("Enhanced image is not a valid numpy array")

def process_source_image(source_img_cv):
    """Process source image to 1040x1435"""
    if source_img_cv is None:
        logger.error("Invalid source image")
        return None

    # Convert to RGB if needed
    if len(source_img_cv.shape) == 2:
        source_img_cv = cv2.cvtColor(source_img_cv, cv2.COLOR_GRAY2RGB)
    elif source_img_cv.shape[2] == 4:
        source_img_cv = cv2.cvtColor(source_img_cv, cv2.COLOR_RGBA2RGB)
    elif source_img_cv.shape[2] == 1:
        source_img_cv = cv2.cvtColor(source_img_cv, cv2.COLOR_GRAY2RGB)

    # Crop if needed (example: crop 20px from sides)
    height, width = source_img_cv.shape[:2]
    if width > 100 and height > 100:  # Only crop if image is large enough
        crop_left = min(20, width//4)
        crop_right = width - min(20, width//4)
        source_img_cv = source_img_cv[:, crop_left:crop_right]

    # Resize to target dimensions (1040x1435)
    target_width = 1040
    target_height = 1435
    resized_img = cv2.resize(source_img_cv, (target_width, target_height), 
                           interpolation=cv2.INTER_LANCZOS4)
    
    logger.info(f"Source image processed to size: {target_width}x{target_height}")
    return resized_img


def create_final_image(swapped_image):
    """Create final 1200x1800 image with frame and logo"""
    # Load frame image (should be 1200x1800)
    frame_path = 'assets/frame.jpg'
    frame_img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    
    if frame_img is None:
        logger.error("Could not load frame image")
        return None
    
    # Load logo image
    logo_path = 'assets/logo11.png'  # Make sure to place your logo in assets folder
    logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    
    if logo_img is None:
        logger.error("Could not load logo image")
        return None
    
    # Resize logo if needed (adjust size as needed)
    logo_height = 150  # Desired logo height
    logo_aspect_ratio = logo_img.shape[1] / logo_img.shape[0]
    logo_width = int(logo_height * logo_aspect_ratio)
    logo_img = cv2.resize(logo_img, (logo_width, logo_height))
    
    # Ensure frame is 1200x1800
    if frame_img.shape[0] != 1800 or frame_img.shape[1] != 1200:
        frame_img = cv2.resize(frame_img, (1200, 1800))
        logger.warning("Frame image was resized to 1200x1800")

    # Resize swapped image to fit within frame (maintaining aspect ratio)
    swapped_height, swapped_width = swapped_image.shape[:2]
    
    # Calculate position to center the swapped image horizontally and move it up
    bottom_margin = 80  # Increased bottom margin
    y_offset = ((1800 - swapped_height) // 2) - bottom_margin
    x_offset = (1200 - swapped_width) // 2
    
    # Calculate logo position (top-right corner)
    logo_margin_right = 130  # Margin from edges
    logo_margin_top = 130
    logo_x = 1200 - logo_width - logo_margin_right
    logo_y = logo_margin_top
    
    # Convert images to RGBA if needed
    if len(frame_img.shape) == 3 and frame_img.shape[2] == 3:
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2RGBA)
    
    if len(swapped_image.shape) == 3 and swapped_image.shape[2] == 3:
        swapped_image = cv2.cvtColor(swapped_image, cv2.COLOR_RGB2RGBA)
    
    # Composite the swapped image
    for y in range(swapped_height):
        for x in range(swapped_width):
            if y + y_offset < 1800 and x + x_offset < 1200:
                if swapped_image[y, x][3] > 0:
                    frame_img[y + y_offset, x + x_offset] = swapped_image[y, x]
    
    # Composite the logo
    for y in range(logo_height):
        for x in range(logo_width):
            if y + logo_y < 1800 and x + logo_x < 1200:
                if logo_img[y, x][3] > 0:  # Check logo transparency
                    frame_img[y + logo_y, x + logo_x] = logo_img[y, x]
    
    return frame_img


def crop_center_width(image):
    """Crop center 60% width of the image and scale up height"""
    if image is None:
        logger.error("Invalid image for cropping")
        return None
    
    # Get image dimensions
    height, width = image.shape[:2]
    logger.info(f"Original image size: {width}x{height}")
    
    # Calculate crop dimensions (60% of width from center)
    crop_width = int(width * 0.7)  # Changed from 0.9 to 0.6 for more aggressive cropping
    start_x = (width - crop_width) // 2
    
    # Crop the image
    cropped_img = image[:, start_x:start_x+crop_width]
    
    # Scale up height while maintaining aspect ratio
    target_height = int(height * 1.4)  # Increase height by 20%
    target_width = int(crop_width * (target_height / height))
    
    # Resize the cropped image
    scaled_img = cv2.resize(cropped_img, (target_width, target_height), 
                          interpolation=cv2.INTER_LANCZOS4)
    
    logger.info(f"Final image size after crop and scale: {target_width}x{target_height}")
    return scaled_img

def perform_face_swap(source_img_base64, custom_prompts=None):
    global result_img_path, result_img_path_firebase
    
    result_img_path = None
    result_img_path_firebase = None
    
    try:
        # Generate target image (1040x1435)
        generate_target_image(custom_prompts)
        
        # Load target image
        target_img_path = 'static/target_image.webp'
        target_img_cv = cv2.imread(target_img_path, cv2.IMREAD_UNCHANGED)
        if target_img_cv is None:
            logger.error("Failed to load target image")
            return
            
        # Process source image to 1040x1435
        source_img_array = np.frombuffer(base64.b64decode(source_img_base64), dtype=np.uint8)
        source_img_cv = cv2.imdecode(source_img_array, cv2.IMREAD_UNCHANGED)
        # source_img_cv = process_source_image(source_img_cv)
        if source_img_cv is None:
            logger.error("Failed to process source image")
            return
            
        # Detect faces
        source_faces = app_insightface.get(source_img_cv)
        target_faces = app_insightface.get(target_img_cv)
        
        if not source_faces or not target_faces:
            logger.error("No faces detected")
            return
            
        # Perform face swap
        source_face = source_faces[0]
        target_face = target_faces[0]
        result_image = swapper.get(target_img_cv, target_face, source_face, paste_back=True)
        
        # Enhance face
        result_image_enhanced = enhance_face(result_image)
        
        # Crop the enhanced image to 60% width
        result_image_cropped = crop_center_width(result_image_enhanced)
        if result_image_cropped is None:
            logger.error("Failed to crop enhanced image")
            return
        
        # Create final composition (1200x1800)
        final_image = create_final_image(result_image_cropped)

        if final_image is None:
            logger.error("Failed to create final image")
            return
            
        # Save result
        unique_filename = generate_unique_filename()
        result_img_path_temp = os.path.join('static', unique_filename)
        cv2.imwrite(result_img_path_temp, final_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        
        # Upload to Firebase
        result_img_url_firebase = upload_image_to_firebase(result_img_path_temp)
        
        # Generate QR code
        if result_img_url_firebase:
            qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
            qr.add_data(result_img_url_firebase)
            qr.make(fit=True)
            qr_code_img = qr.make_image(fill_color="black", back_color="white")
            qr_code_path = os.path.join('static', 'qr_code.png')
            qr_code_img.save(qr_code_path, format="PNG")
        
        with result_lock:
            result_img_path = result_img_path_temp
            result_img_path_firebase = result_img_url_firebase
            
    except Exception as e:
        logger.error(f"Error in perform_face_swap: {str(e)}")

# Flask routes remain the same as in your original code
@app.route('/')
def main():
    return render_template('main.html')

@app.route('/index', methods=['POST'])
def index():
    return render_template('index.html')

@app.route('/swap', methods=['POST'])
def swap_faces():
    source_img_base64 = request.form['source']
    custom_prompts = request.form.get('prompt')
    
    # Log source image info
    source_img_array = np.frombuffer(base64.b64decode(source_img_base64), dtype=np.uint8)
    source_img_cv = cv2.imdecode(source_img_array, cv2.IMREAD_UNCHANGED)
    if source_img_cv is not None:
        height, width = source_img_cv.shape[:2]
        logger.info(f"Received source image size: {width}x{height} pixels")
    else:
        logger.error("Failed to decode source image")

    thread = threading.Thread(target=perform_face_swap, args=(source_img_base64, custom_prompts))
    thread.start()
    return render_template('loading.html')

@app.route('/check_status', methods=['GET'])
def check_status():
    global result_img_path, result_img_path_firebase
    
    if result_img_path or result_img_path_firebase:
        return render_template('result.html', 
                            name="", 
                            result_img_path_firebase=result_img_path_firebase, 
                            result_img_path=result_img_path)
    
    time.sleep(1)
    return render_template('loading.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
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
import firebase_admin # type: ignore
from firebase_admin import credentials, storage # type: ignore
import time
import logging
import datetime
import qrcode
import time
from io import BytesIO
import shutil
from gfpgan import GFPGANer

app = Flask(__name__)

# Disable caching for specific routes
@app.after_request
def add_header(response):
    # Disable caching for the specified routes
    if request.path == '/static/target_image.png' or request.path == '/static/qr_code.png':
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# Initialize Firebase with your credentials
cred = credentials.Certificate('./serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'ai-photo-7495c.appspot.com'
})

# Get a reference to the Firebase Storage bucket
bucket = storage.bucket()

app_insightface = FaceAnalysis(name='buffalo_l')
app_insightface.prepare(ctx_id=0, det_size=(640, 640))

model_path = "./inswapper_128.onnx"  # Replace with the actual path
swapper = insightface.model_zoo.get_model(model_path)

# Global variable to store the result image path
result_img_path = None
result_lock = threading.Lock()

# Function to generate a unique filename
def generate_unique_filename():
    return str(uuid.uuid4()) + '.png'

# Function to generate the target image using AI
def generate_target_image(custom_prompts=None):
    ##5940bd855265365e5490344577a80089bc984dfb216a0c2a7a17644e2e6f0cbd48dcb92acd8917122137382dea77493a
    # Replace 'YOUR_API_KEY' with your actual Clipdrop API key
    clipdrop_api_key = '5940bd855265365e5490344577a80089bc984dfb216a0c2a7a17644e2e6f0cbd48dcb92acd8917122137382dea77493a'

    predefined_prompts_str = "photorealistic concept art, high quality digital art, cinematic, hyperrealism, photorealism, Nikon D850, 8K., sharp focus, emitting diodes, artillery, motherboard, by pascal blanche rutkowski repin artstation hyperrealism painting concept art of detailed character design matte painting, 4 k resolution"

    all_prompts = predefined_prompts_str
    if custom_prompts:
        all_prompts += "\n" + custom_prompts
        print(all_prompts)
    # Define the Clipdrop API endpoint URL
    clipdrop_url = 'https://clipdrop-api.co/text-to-image/v1'

    # Set up the request headers with your Clipdrop API key
    headers = {
        'x-api-key': clipdrop_api_key,
        'accept': 'image/webp',
        'x-clipdrop-width': '400',  # Desired width in pixels
        'x-clipdrop-height': '600',  # Desired height in pixels
    }

    # Create a dictionary with the prompt data
    data = {
        'prompt': (None, all_prompts, 'text/plain')
    }

    # Send a POST request to the Clipdrop API
    response = requests.post(clipdrop_url, files=data, headers=headers)

    if response.ok:
        # Save the image content to a file in the 'static' folder
        with open('static/target_image.webp', 'wb') as f:
            f.write(response.content)

    else:
        # Handle the case when image generation fails
        pass

result_img_path_firebase = None

# Function to upload an image to Firebase Storage and generate a URL with token
def upload_image_to_firebase(result_img_path_temp):
    # Generate a unique filename for the swapped image
    unique_filename = generate_unique_filename()

    try:
        # Upload the result image to Firebase Storage
        blob = bucket.blob('ig-ai-images/' + unique_filename)
        blob.upload_from_filename(result_img_path_temp)

                # Save a copy of the image locally with the name "print.png"
        print_img_path_local = os.path.join('static', 'print.png')
        # Copy the result image to the local path
        shutil.copyfile(result_img_path_temp, print_img_path_local)

        # Get the URL of the uploaded image with a token
        expires_in = datetime.timedelta(days=1)  # Set an expiration time for the token
        url = blob.generate_signed_url(expires_in)

        return url

    except Exception as e:
        return None
    
    
gfpganer = GFPGANer(model_path='GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2)
def enhance_face(image):
    logging.info("Starting face enhancement...")
    _, _, restored_img = gfpganer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
    
    logging.info(f"Type of restored_img: {type(restored_img)}")
    if isinstance(restored_img, Image.Image):
        restored_img = np.array(restored_img)
    logging.info(f"Type after conversion (if any): {type(restored_img)}")
    if isinstance(restored_img, np.ndarray):
        logging.info("Face enhancement completed.")
        return restored_img
    else:
        raise ValueError("Enhanced image is not a valid numpy array")
    
    
    
def perform_face_swap(source_img_base64, custom_prompts=None):
    global result_img_path, result_img_path_firebase

    # Clear the previous result image paths
    result_img_path = None
    result_img_path_firebase = None

    # Generate the target image using the AI
    generate_target_image(custom_prompts)

    # Load the selected target image
    target_img_path = 'static/target_image.webp'  # Modify this path accordingly
    with open(target_img_path, 'rb') as target_file:
        target_img_data = target_file.read()

    target_img_cv = cv2.imdecode(np.frombuffer(target_img_data, dtype=np.uint8), -1)

    # Decode the Base64-encoded source image
    source_img_array = np.frombuffer(base64.b64decode(source_img_base64), dtype=np.uint8)
    source_img_cv = cv2.imdecode(source_img_array, -1)

    # Ensure that the source image has 3 color channels (remove alpha channel if present)
    if source_img_cv.shape[2] == 4:
        source_img_cv = cv2.cvtColor(source_img_cv, cv2.COLOR_RGBA2RGB)

    # Detect faces in the source and target images
    source_faces = app_insightface.get(source_img_cv)
    target_faces = app_insightface.get(target_img_cv)

    # Check if at least one face is detected in each image
    if not source_faces or not target_faces:
        return

    # Use the first detected face
    source_face = source_faces[0]
    target_face = target_faces[0]

    # Perform face swapping
    result_image = swapper.get(target_img_cv, target_face, source_face, paste_back=True)

    # Enhance the swapped face (result_image)
    result_image_enhanced = enhance_face(result_image)

    # Load the frame image
    frame_path = 'assets/frame.jpg'
    frame_img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

    # Resize the result image to fit within the frame
    frame_height, frame_width = frame_img.shape[:2]
    # result_image_resized = cv2.resize(result_image_enhanced, (frame_width, frame_height))
    result_height, result_width = result_image_enhanced.shape[:2]
    # Calculate the position to center the result image on the frame
    y_offset = int((frame_height - result_image.shape[0]) * 0.3)
    x_offset = int((frame_width - result_image.shape[1]) * 0.5)

    # Overlay the result image onto the frame
     # Overlay the result image onto the frame without changing the aspect ratio
    for c in range(0, 3):
        frame_img[y_offset:y_offset+result_height, x_offset:x_offset+result_width, c] = \
            result_image_enhanced[:, :, c]


    # Save and upload the final enhanced image with frame
    unique_filename = generate_unique_filename()
    result_img_path_temp = os.path.join('static', unique_filename)
    cv2.imwrite(result_img_path_temp, frame_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 95])

    # Upload the final image to Firebase Storage
    result_img_url_firebase = upload_image_to_firebase(result_img_path_temp)
    print(result_img_url_firebase)

    # Generate a QR code for the Firebase URL
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(result_img_url_firebase)
    qr.make(fit=True)

    # Create and save the QR code image
    qr_code_img = qr.make_image(fill_color="black", back_color="white")
    qr_code_path = os.path.join('static', 'qr_code.png')
    qr_code_img.save(qr_code_path, format="PNG")

    with result_lock:
        result_img_path = result_img_path_temp
        result_img_path_firebase = result_img_url_firebase  # Store Firebase URL with token

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/index', methods=['POST'])
def index():
    name = request.form['name']
    return render_template('index.html', name=name)

@app.route('/swap', methods=['GET','POST'])
def swap_faces():
    # Get the source image Base64 data from the form
    source_img_base64 = request.form['source']

    # Get the custom prompts from the form
    custom_prompts = request.form.get('prompt')

    # Use a thread to perform the face swap in the background
    thread = threading.Thread(target=perform_face_swap, args=(source_img_base64, custom_prompts))
    thread.start()

    return render_template('loading.html')

#working one
@app.route('/check_status', methods=['GET'])
def check_status():
    global result_img_path, result_img_path_firebase

    if result_img_path or result_img_path_firebase:
        print(result_img_path_firebase)
        # Image generation is complete, redirect to the result page
        return render_template('result.html', name="", result_img_path_firebase=result_img_path_firebase, result_img_path=result_img_path)
    
    time.sleep(15) 
    return render_template('loading.html')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
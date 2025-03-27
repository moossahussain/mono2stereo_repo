from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from inference_pipeline import run_inference

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    """
    Renders the home page containing the upload form.

    Returns:
        str: Rendered HTML template for the homepage (index.html).
    """
    return render_template('index.html')

@app.route('/generate-3d', methods=['POST'])
def generate_3d():
    """
    Handles POST requests for generating 3D view of an uploaded image.

    Receives an image and model type from the frontend, processes the image
    using the inference pipeline, and returns file paths and depth stats
    as a JSON response.

    Returns:
        Response: JSON object containing paths to processed images and
        depth statistics including center, average, and top-left region depth.
    """
    file = request.files['image']
    model_type = request.form.get('model', 'midas')
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    outputs = run_inference(image_path, OUTPUT_FOLDER, model_type)
    base_name = os.path.splitext(file.filename)[0]  # for stereo view URL

    return jsonify({
        "detection": f"/output/{os.path.basename(outputs['detection'])}",
        "segmentation": f"/output/{os.path.basename(outputs['segmentation'])}",
        "depth": f"/output/{os.path.basename(outputs['depth'])}",
        "right": f"/output/{os.path.basename(outputs['right'])}",
        "stereo": f"/stereo-view/{base_name}",  # for WebXR
        "stats": outputs["depth_stats"]
    })

@app.route('/output/<filename>')
def output_file(filename):
    """
    Serves a processed output file (e.g., detection, depth map, right view).

    Args:
        filename (str): Name of the file to be served.

    Returns:
        Response: File from the OUTPUT_FOLDER.
    """
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/uploads/<filename>')
def upload_file(filename):
    """
    Serves an uploaded original image file.

    Args:
        filename (str): Name of the uploaded file.

    Returns:
        Response: File from the UPLOAD_FOLDER.
    """
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/stereo-view/<filename>')
def stereo_view(filename):
    """
    Renders the stereo view page for WebXR visualization.

    Args:
        filename (str): Base name of the uploaded image (without extension).

    Returns:
        str: Rendered stereo.html template with left and right eye image URLs.
    """
    return render_template(
        'stereo.html',
        left=f"/uploads/{filename}.png",
        right=f"/output/{filename}_right.jpg"
    )

if __name__ == '__main__':
    """
    Entry point for running the Flask app in debug mode.
    """
    # app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=5000)


import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

app = FastAPI()

# Load the pre-trained model
model = load_model('demo1.h5')

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

IMAGE_SIZE = 150  # Assuming images are resized to 150x150 for model input

# Function to detect anomaly based on mean pixel intensity
def detect_anomaly(img_array):
    img_data = img_array[0]
    mean_intensity = np.mean(img_data, axis=(0, 1))  # Shape: (3,) for RGB channels
    intensity_threshold = 0.8  # Adjust this threshold based on your data characteristics

    if np.any(mean_intensity < intensity_threshold) or np.any(mean_intensity > 1.0):
        anomaly_detected = True
    else:
        anomaly_detected = False

    if anomaly_detected:
        bounding_box = (20, 20, 130, 130)
    else:
        bounding_box = None

    return anomaly_detected, bounding_box

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).resize((IMAGE_SIZE, IMAGE_SIZE))

        # Convert to numpy array and normalize
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Adjust the image shape to match model input
        if img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)  # Convert to 3-channel image

        # Make a prediction
        prediction = model.predict(img_array)
        class_id = int(np.round(prediction[0][0]))

        # Determine class label and color
        if class_id == 0:
            class_label = "benign"
            box_color = "yellow"
        elif class_id == 1:
            class_label = "malignant"
            box_color = "red"
        else:
            class_label = "Error: Prediction out of bounds"
            box_color = "darkgreen"

        anomaly_detected, bounding_box = detect_anomaly(img_array)

        draw = ImageDraw.Draw(img)
        if anomaly_detected and bounding_box:
            # draw.rectangle(bounding_box, outline=box_color, width=3)
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = ImageFont.load_default()
            # text_position = (bounding_box[0], bounding_box[1] - 20)
            # draw.text(text_position, f"{class_label}", fill=box_color, font=font)

        result_img_path = "app/static/result_image.png"
        img.save(result_img_path)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "class_id": class_id,
            "class_label": class_label,
            "image_url": "/static/result_image.png",
            "error": False  # Flag indicating no error occurred
        })
    except Exception as e:
        print(f"Exception occurred: {e}")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error_message": "An internal server error occurred. Please try again later.",
            "error": True  # Flag indicating error occurred
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
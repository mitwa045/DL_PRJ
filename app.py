
import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import numpy as np

# Load your model (make sure the model file is accessible)
model = load_model('DN1_.h5')

# Function to convert to ELA image
def convert_to_ela_image(image, quality): 
    ela_filename = 'temp_ela.png'
    image = image.convert('RGB')
    
    # Save the original image to a temporary file for ELA comparison
    temp_filename = 'temp_file_name.jpg'
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = sum([ex[1] for ex in extrema]) / 3
    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

# Function to prepare input image
def prepare_image(image):
    ela_image = convert_to_ela_image(image, 90)  # Use the image directly
    return np.array(ela_image.resize((128, 128))) / 255.0  # Resize and normalize

# Function to make predictions
def predict_image(img):
    try:
        # Directly prepare the image from the uploaded PIL Image
        preprocessed_image = prepare_image(img)
        preprocessed_image = preprocessed_image.reshape(1, 128, 128, 3)
        
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions)
        return "Prediction: Authentic Image" if predicted_class == 0 else "Prediction: Tampered Image"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Tampering Detection",
    description="Upload an image to check for tampering."
)

# Launch the Gradio interface
interface.launch(share=True)

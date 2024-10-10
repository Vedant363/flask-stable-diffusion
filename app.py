from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from dotenv import load_dotenv
import google.generativeai as genai
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)
load_dotenv()

# Set a secret key for session management (fix for the error)
app.secret_key = os.getenv('SECRET_KEY', 'VEDANT')

# Load the API key
api_key = os.getenv('API_KEY')
if not api_key:
    raise RuntimeError("API_KEY not found in environment variables.")
genai.configure(api_key=api_key)

# Initialize the Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Set the model ID for Stable Diffusion
model_id = "stabilityai/stable-diffusion-2-1"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Step 1: Collect user inputs from the form
            poster_theme = request.form['poster_theme']
            color_scheme = request.form['color_scheme']
            art_direction = request.form['art_direction']
            poster_text = request.form['poster_text']
            poster_orientation = request.form['poster_orientation']
            use_case = request.form['use_case']
            poster_resolution = request.form['poster_resolution']
            additional_effect = request.form['additional_effect']

            # Combine inputs into a single prompt
            user_input = (
                f"{poster_theme} theme, {color_scheme} color scheme, {art_direction} art direction, "
                f"containing the text '{poster_text}', {poster_orientation} orientation, for {use_case}, "
                f"resolution {poster_resolution}, with additional effect of {additional_effect}"
            )

            # Step 2: Refine the user input using the model with more focus on 'poster'
            response = model.generate_content(
                f"Generate a one-line prompt to create a **poster** with the following details: {user_input}. The emphasis should be on creating a visually striking poster."
            )
            refined_prompt = response.text
            print(f"Refined Prompt: {refined_prompt}")

            # Step 3: Generate an image with Stable Diffusion
            image = pipe(refined_prompt).images[0]

            # Step 4: Save the generated image
            image_path = "static/generated_image.png"
            image.save(image_path)

            flash('Image generated successfully!', 'success')
            return redirect(url_for('result', image_path=image_path))
        except Exception as e:
            flash(f"An error occurred: {str(e)}", 'danger')
            return redirect(url_for('home'))

    return render_template('index.html')

@app.route('/result')
def result():
    image_path = request.args.get('image_path')
    return render_template('result.html', image_path=image_path)

if __name__ == '__main__':
    app.run()

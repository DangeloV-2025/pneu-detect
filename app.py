from flask import Flask, render_template, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import pickle

import numpy
import torch
from torchvision import transforms, models
from PIL import Image
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F



app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# load model
resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

num_features = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_features, 2)


resnet.load_state_dict(torch.load("models/models_jul12/resnet18_epoch_8.pth"))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        
        image = Image.open(filepath).convert('RGB')
        print(type(image))

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        print(type(image_tensor))
        resnet.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient calculation
            output = resnet(image_tensor)
            

            print(type(output))

        # Assuming output is logits or scores, apply sigmoid for probability
        #probability = torch.sigmoid(output)
        
        # Convert probability tensor to a scalar value
        #predicted_class = (probability > 0.5).int().item()

        # Debugging output
        print(f"Model output shape: {output.shape}")
        
        # print(f"Model output values: {output}")
        #print(f"Predicted class: {predicted_class}")

        # prediciton = torch.sigmoid(1- output[0].int())


        predicted_class = torch.argmax(output).item()

        if (predicted_class == 1):
            prediction = "Patient has Pneumonia"
        elif(predicted_class == 0):
            prediction = "Patient does not have Pneumonia"
        else:
            prediction = "Error in processing image"



        


        # Convert probability tensor to a scalar value
        #predicted_class = (probability > 0.5).int().item()

        #predicted_class = torch.mean(output, dim=1)
        

        return render_template('index.html', filename=filename, prediction=prediction)
    else:
        return redirect(request.url)


@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

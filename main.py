from fastapi import FastAPI, UploadFile, File
from model_runner import load_model, predict_class, get_class_label
from PIL import Image
import torchvision.transforms as transforms

app = FastAPI()

# Load the model and label map
model = load_model('model.pth')


transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

import os
import io
from PIL import Image

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     # Save the uploaded file to disk
#     with open(file.filename, "wb") as buffer:
#         buffer.write(await file.read())

#     # Check that the file exists and is readable
#     if not os.path.isfile(file.filename) or not os.access(file.filename, os.R_OK):
#         return {"error": "File not found or not readable"}

#     # Open the image file using Pillow
#     with io.BufferedReader(io.FileIO(file.filename, 'rb')) as f:
#         image = Image.open(f)

#     # Make a prediction
#     predicted_class = predict_class(image, model, transform)
#     class_label = get_class_label(predicted_class)

#     # Return the predicted class name
#     return {"class": class_label}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    print(type(file))

    ## Save the uploaded file to disk
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())

    # Check that the file exists and is readable
    if not os.path.isfile(file.filename) or not os.access(file.filename, os.R_OK):
        return {"error": "File not found or not readable"}
    
    # Open the image file using Pillow
    with io.BufferedReader(io.FileIO(file.filename, 'rb')) as f:
        image = Image.open(f)

    ## save the image in data if it is not already there
    if not os.path.isfile('data/'+file.filename):
        image.save('data/'+file.filename)

    # get the image path
    image_path = 'data/'+file.filename

    print(image_path)

    # Make a prediction
    predicted_class = predict_class(image_path, model, transform)
    class_label = get_class_label(predicted_class)
    print(f"Image '{image_path}' predicted as: {class_label}")

    # with open(file.filename, "wb") as buffer:
    #     buffer.write(await file.read())

    # # Check that the file exists and is readable
    # if not os.path.isfile(file.filename) or not os.access(file.filename, os.R_OK):
    #     return {"error": "File not found or not readable"}

    # # Open the image file using Pillow
    # with io.BufferedReader(io.FileIO(file.filename, 'rb')) as f:
    #     image = Image.open(f)

    # predicted_class = predict_class(image, model, transform)
    # class_label = get_class_label(predicted_class)
    # #return {"filename": file.filename}
    # return {"class": class_label}



# app.py
import sys

from click import style

sys.path.append("../")
from io import BytesIO
import base64
import requests
import torch
import os
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network.Transformer import Transformer
from tqdm import tqdm_notebook
from test_from_code import transform
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os

import boto3
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret-key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

with open("secret.txt") as f:
    data = f.readlines()
aws_accesskey = data[0].strip('\n')
aws_secretkey = data[1].strip('\n')

# print(aws_accesskey,aws_secretkey)

client = boto3.client(
    's3',
    aws_access_key_id=aws_accesskey,
    aws_secret_access_key=aws_secretkey,

)
print(client)

s3 = boto3.resource('s3')

for bucket in s3.buckets.all():
    print(bucket.name)

def img_to_base64_str(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str


def load_models(s3, bucket):

    styles = ["Hosoda", "Hayao", "Shinkai", "Paprika"]
    models = {}

    for style in styles:
        model = Transformer()
        response = s3.get_object(Bucket=bucket, Key=f"models/{style}_net_G_float.pth")
        state = torch.load(BytesIO(response["Body"].read()))
        model.load_state_dict(state)
        model.eval()
        models[style] = model

    return models


gpu = -1

s3 = boto3.client("s3")
bucket = "cartoonmini"

mapping_id_to_style = {0: "Hosoda", 1: "Hayao", 2: "Shinkai", 3: "Paprika"}

models = load_models(s3, bucket)
print(f"models loaded ...")
path = 'test_img/7--136.jpg'


img = Image.open(path)
with open(path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')


data = {
    "image": encoded_string,
    "model_id": 1,
    "load_size": 500

}
# print(data)




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/edit')
def edit():
    return render_template('edit.html')




@app.route('/', methods=['POST'])
def form_data():
    load_size=400

    if request.method=="POST":
        style=request.form["filter"]
        print(style)

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        flash(file.filename)
        print(file.filename)

        path = UPLOAD_FOLDER + file.filename
        print(path)
        output = transform(models, style, path, load_size)
        output.save('static/output/image_output.jpg')
        return render_template('index.html', filename=filename)

    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        flash(file.filename)
        print(file.filename)

        return render_template('index.html', filename=filename)

    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    print(filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)



# @app.route('/edit', methods=['GET'])
# def edited_image(filename):
#     return redirect(url_for('edit.html', filename='output/image_output.jpg'), code=301)



if __name__ == "__main__":
    app.run()

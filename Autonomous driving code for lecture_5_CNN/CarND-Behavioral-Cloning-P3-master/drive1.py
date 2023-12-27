import argparse
import base64
from datetime import datetime
import os
import shutil


import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torchvision
from torchvision.io import read_image
from torchvision import models, transforms
import glob
import os
import csv
from torch import optim

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = data["steering_angle"]
        speed = data["speed"]
        throttle = data["throttle"]

        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        trf = transforms.Compose([  transforms.ToTensor(),
                                    transforms.Resize((224,224)),
            ])
        image = trf(image).to('cuda')
        steering_angle = float(model(image).item())
        throttle = controller.update(float(speed))
        print(steering_angle)


        print(steering_angle)
        send_control(steering_angle,throttle)

       
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.convn_1 = nn.Conv2d(3,64,(2,2))
        self.convn_2 = nn.Conv2d(64,128,(2,2))
        self.convn_3 = nn.Conv2d(128,256,(2,2))
        self.max_pool = nn.MaxPool2d((3,3))
        self.layer_1 = nn.Linear(12544,1000)
        self.layer_2 = nn.Linear(1000,1000)
        self.layer_3 = nn.Linear(1000,100)
        self.layer_4 = nn.Linear(100,1)
    
    def forward(self, x):
        x = self.convn_1(x)
        x = self.max_pool(x)
        x = self.convn_2(x)
        x = self.max_pool(x)
        x = self.convn_3(x)
        x = self.max_pool(x)
        x = x.view((-1,12544))
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        return x


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0,0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version


    model = Net()
    model = model.to('cuda')
    model.load_state_dict(torch.load(r'C:\Users\monsi\Desktop\ALL folders\club\lecture_5_CNN\CarND-Behavioral-Cloning-P3-master\my_model_1.pt'))
    model.eval()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

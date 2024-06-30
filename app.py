from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re

import pickle
import numpy as np # linear algebra
import pandas as pd

import joblib
import cv2


# Flask utils
from flask import Flask, redirect, url_for, request, render_template


import pickle
import sqlite3
import random

import smtplib 
from email.message import EmailMessage
from datetime import datetime


app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_path2 = pickle.load(open('model.pkl','rb'))
CTS = model_path2

  
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')


@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "evotingotp4@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("evotingotp4@gmail.com", "xowpojqyiygprhgr")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict_lo', methods=['POST'])
def predict_lo():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("signin.html")

@app.route('/home')
def home():
	return render_template('home.html')


@app.route('/predict2',methods=['GET','POST'])
def predict2():
    #pred = []
    print("Entered")
    
    print("Entered here")
    file = request.files['files'] # fet input
    val1, val2, val3, val4, val5, val6, val7,val8,val9,val10,val11,val12,val13 = (request.form['0']),(request.form['1']), (request.form['2']), (request.form['3']), (request.form['4']), (request.form['5']), (request.form['6']), (request.form['7']), (request.form['8']), (request.form['9']), (request.form['10']), (request.form['11']), (request.form['12'])

    filename = file.filename        
    print("@@ Input posted = ", filename)
        
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    image = cv2.imread(file_path)
    image = cv2.resize(image , (32,32))
    
    image = image.reshape([-1, np.product((32,32,3))])
    result = CTS.predict(image)


    final1 = np.array([val1, val2, val3, val4, val5, val6, val7,val8,val9,val10,val11,val12,val13]).reshape(1,-1)
    model = joblib.load("model.sav")
    predict = model.predict(final1)
    predict = predict[0]
    print(predict)
    #bili_level calculation
    size_x = 128
    size_y = 128
    OGX = 3000
    OGY = 1700
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    test_image = test_img 
    test_img = cv2.resize(test_img, (size_x, size_y))
        

    test_img = np.expand_dims(test_img, axis=0)

    prediction = model.predict(test_img)

    test_image = cv2.resize(test_image, (OGX, OGY))
    prediction_image = prediction.reshape((size_x,size_y))

    prediction_image = cv2.resize(prediction_image, (OGX, OGY))
    prediction_image=cv2.imread('/images/Myfolder/segmented.jpg')
    imgg = cv2.resize(prediction_image, (OGX, OGY))
    annd = cv2.bitwise_and(imgg, test_image)
    image = cv2.imread('/images//Myfolder/segmented.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    arr = cv2.resize(image, (32, 32))
    for j in range(arr.shape[1]):
        column = arr[:, j]
        non_zero_elements = column[column != 0]
        average = np.mean(non_zero_elements) if non_zero_elements.size > 0 else 0
        arr[:, j][arr[:, j] == 0] = average

    CIEXYZ = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)

    normalized_CIEXYZ = cv2.normalize(CIEXYZ, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    X, Y, Z = cv2.split(normalized_CIEXYZ)
    a1 = -11.851
    a2 = -0.031
    a3 = 0.183
    a4 = -0.153
    a5 = 0.4

    R,G,B=arr[:,0],arr[:,1],arr[:,2]
    mean_R = np.mean(R[R != 0]).astype(int)
    mean_G = np.mean(G[G != 0]).astype(int)
    mean_B = np.mean(B[B != 0]).astype(int)
    mean_Y = np.mean(YI).astype(int)
    bili_level = a1 + a2 * mean_R + a3 * mean_G + a4 * mean_B + a5 * mean_Y
    if result == 0 and predict == 1 and bili_level >3:
        pred = "The Patient is Diagnosis with Jaundice with bilirubin level "+(bili_level)
    elif result == 1 and predict == 0 and bili_level <3:
        pred = "The Patient is Not Diagnosis with Jaundice with bilirubin level"+str(bili_level)
    elif result == 0 and predict == 0 and bili_level >3:
        pred = "The Patient is Diagnosis with Jaundice based on Eye Image! with bilirubin level"+str(bili_level)
    elif result == 1 and predict == 1 and bili_level <3:
        pred = "The Patient is Not Diagnosis with Jaundice based on Eye Image, But Diagnosis with Jaundice based on Data Provide, Suggesting for Doctor Consulation! with bilirubin level"+str(bili_level)

    return render_template('after.html', pred_output = pred, img_src=UPLOAD_FOLDER + file.filename,)

@app.route('/about')
def about():
	return render_template('graph.html')

@app.route('/notebook1')
def notebook1():
	return render_template('Image.html')

@app.route('/notebook2')
def notebook2():
	return render_template('Data.html')


if __name__ == '__main__':
    app.run(debug=False)
from flask import Flask, render_template, request, redirect, url_for, session
import cv2
import numpy as np
import os
import pandas as pd
from dotenv import load_dotenv
import empyrebase

config = {
    "apiKey": os.getenv('API_KEY'),
    "authDomain": os.getenv('AUTH_DOMAIN'),
    "projectId": os.getenv('PROJECT_ID'),
    "storageBucket": os.getenv('STORAGE_BUCKET'),
    "messagingSenderId": os.getenv('MESSAGING_SENDER_ID'),
    "appId": os.getenv("APP_ID"),
    "measurementId": os.getenv('MEASUREMENT_ID'),
    "databaseURL": os.getenv('DATABASE_URL')
}

firebase = empyrebase.initialize_app(config)
auth = firebase.auth()

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.getenv('SECRET_KEY')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    if 'email' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['uid'] = user['idToken']
            return redirect(url_for('dashboard'))
        except:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        omr = request.files['omr']
        excel = request.files['excel']
        output_path_omr = os.path.join(app.config['UPLOAD_FOLDER'], omr.filename)
        output_path_excel = os.path.join(app.config['UPLOAD_FOLDER'], excel.filename)
        omr.save(output_path_omr)
        excel.save(output_path_excel)
        result, total = process_omr(output_path_omr, output_path_excel)
        return render_template("result.html", result=result.items(), total=total)

    if 'uid' in session:
        return render_template('dashboard.html')
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        user = auth.create_user_with_email_and_password(email, password)

        session['uid'] = user['idToken']
        return redirect(url_for('dashboard'))

    return render_template("register.html")

def process_omr(omr_path, answer_path):
    img = cv2.imread(omr_path)
    imgContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(img, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)
    imgInvert = cv2.bitwise_not(imgGray)

    th, threshed = cv2.threshold(imgGray, 115, 255,cv2.THRESH_BINARY_INV)
    thresh = threshed

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    result = img.copy() 
    centers = []
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    min_area = 190 
    bubbles = [c for c in contours if cv2.contourArea(c) > min_area]
    bubbles = sorted(bubbles, key=lambda c: cv2.boundingRect(c)[0])


    i = 1
    for cntr in bubbles:
        M = cv2.moments(cntr)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx,cy))
        cv2.circle(result, (cx, cy), 10, (0, 255, 0), -1)
        pt = (cx,cy)
        i = i + 1



    rows = 20
    cols = 4
    choices = 4
    topics = ['PYTHON', 'DATA ANALYSIS', 'MySQL', 'POWER BI', 'Adv STATS']

    bubble_data = {}

    bubble_width = thresh.shape[1] // cols
    bubble_height = thresh.shape[0] // rows
    bubbles_per_row = len(bubbles) // rows

    for i, bubble in enumerate(bubbles):
        if i==99:
            break
        x, y, w, h = cv2.boundingRect(bubble)

        row = i // cols
        col = i % cols
        q_num = row * cols + col + 1
        topic = topics[(q_num - 1) // 20] 
        
        roi = thresh[y:y + h, x:x + w]

        bubble_w = w // choices
        filled = None

        for j in range(choices):
            bx = x + j * bubble_w
            by = y
            bubble_section = roi[:, j * bubble_w:(j + 1) * bubble_w]
            
            total_filled = cv2.countNonZero(bubble_section)

            if filled is None or total_filled > filled[0]:
                filled = (total_filled, chr(65 + j)) 

        bubble_data[q_num] = {'topic': topic, 'answer': filled[1]}


    answers = process_answers(answer_path)

    result = {}

    for key, value in bubble_data.items():
        if answers[key-1] == value['answer'].lower():
            if value['topic'] not in result.keys():
                result[value['topic']] = 1
            else:
                result[value['topic']] += 1

    total = 0
    for x,y in result.items():
        total += y

    return result, total

def process_answers(answer_key_path):
    excel_file = pd.ExcelFile(answer_key_path)

    sheets = excel_file.sheet_names
    answers = []

    for sheet_name in sheets:
        df = excel_file.parse(sheet_name)

        for topic_column in df.columns:
            for index, value in df[topic_column].dropna().items():
                try:
                    answers.append(value.split('-')[1].strip())
                except:
                    answers.append(value.split('.')[1].strip())

    return answers


if __name__ == '__main__':

    app.run(debug=True)



import pickle
from flask import *
import firebase_admin
from firebase_admin import credentials, firestore, auth
import pandas as pd
import torch
from torch import nn
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
nltk.download('punkt')
stemmer= SnowballStemmer(language= 'english')
nltk.download('stopwords')

app = Flask(__name__)

cred = credentials.Certificate("firebase_key.json")  
firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)

name = ''
passw = ''
mail = ''
mobile = ''

model1 = pickle.load(open("rf.pkl", "rb"))

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        age = request.form['age']
        gender = request.form['sex']
        polyuria = request.form['polyuria']
        polydipsia = request.form['polydipsia']
        s_w_l = request.form['swl']
        weakness = request.form['weakness']
        polyphagia = request.form['polyphagia']
        g_t = request.form['gt']
        v_b = request.form['vb']
        itching = request.form['itching']
        irritability = request.form['irritability']
        d_h = request.form['dh']
        p_p = request.form['pp']
        m_s = request.form['ms']
        alopecia = request.form['alopecia']
        obesity = request.form['obesity']
        df = pd.DataFrame(data=[[gender, polyuria, polydipsia, s_w_l, weakness, polyphagia, g_t, v_b, itching, irritability, d_h, p_p, m_s, alopecia, obesity, age]],
                          columns=['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
                                   'Itching', 'Irritability', 'delayed healing', 'partial paresis',
                                   'muscle stiffness', 'Alopecia', 'Obesity', 'age_group'])
        df['age_group'].replace(['10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90'], [0, 1, 2, 3, 4, 5, 6, 7], inplace=True)
        df['Gender'].replace(['Female', 'Male'], [0, 1], inplace=True)
        df['Polyuria'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['Polydipsia'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['sudden weight loss'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['weakness'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['Polyphagia'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['Genital thrush'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['visual blurring'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['Itching'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['Irritability'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['delayed healing'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['partial paresis'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['muscle stiffness'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['Alopecia'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['Obesity'].replace(['No', 'Yes'], [0, 1], inplace=True)
        prediction = model1.predict(df)
        if prediction == 0:
            msg = 'You are tested negative!!!!'
            return render_template('prediction2.html', msg=msg)
        else:
            msg = 'You may be diabetic!!!! Please consult doctor for further process!!!'
            return render_template('prediction.html', msg=msg)
    return render_template('diabetes.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            users_ref = db.collection('users')
            query = users_ref.where('email', '==', email).limit(1)
            results = query.stream()
            user_data = list(results)
            if not user_data:
                msg = "User doesn't exist! Please register!"
                return render_template('signup.html', msg=msg)
            else:
                user_doc = user_data[0].to_dict()
                if user_doc['password_user'] == password:
                    name = user_doc['username']
                    return render_template('homepage.html', content=name)
                else:
                    msg = 'Incorrect password. Try again!'
        except Exception as e:
            msg = f'Error: {str(e)}'

    return render_template('login.html', msg=msg)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form and 'mobilenumber' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        mobilenumber = request.form['mobilenumber']
        try:
            user = auth.create_user(
                email=email,
                password=password,
                display_name=username
            )
            db.collection('users').document(user.uid).set({
                'username': username,
                'email': email,
                'password_user': password,
                'mobilenumber': mobilenumber
            })
            name = username
            mail = email
            resp = make_response(redirect(url_for('homepage')))
            return resp
        except Exception as e:
            msg = 'User already exists or error creating user!'
            return render_template('signup.html', msg=msg)
    return render_template('signup.html', msg=msg)

@app.route('/homepage')
def homepage():
    return render_template('homepage.html', content=name)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')



# Hardcoded appointment data
appointments = [
    {"drid": 1, "drname": "Dr. Smith", "dremail": "smith@hospital.com", "Specialist": "Cardiologist", "drstatus": "Available", "booking_status": "Not Booked"},
    {"drid": 2, "drname": "Dr. Johnson", "dremail": "johnson@hospital.com", "Specialist": "Dermatologist", "drstatus": "Available", "booking_status": "Not Booked"},
    {"drid": 3, "drname": "Dr. Lee", "dremail": "lee@hospital.com", "Specialist": "Neurologist", "drstatus": "Not Available", "booking_status": "Booked"},
    {"drid": 4, "drname": "Dr. Allen", "dremail": "allen@hospital.com", "Specialist": "Pediatrician", "drstatus": "Available", "booking_status": "Not Booked"}
]

@app.route('/appointment', methods=['GET'])
def appointment():
    return render_template('appointment.html', info_table=appointments)

@app.route('/appoint', methods=['GET', 'POST'])
def appoint():
    msg = ''
    if request.method == 'POST':
        full_name = request.form['name']
        user_email = request.form['email']
        user_phone = request.form['phone']
        user_date = request.form['date']
        user_time = request.form['time']
        user_special = request.form['specialization']

        # Find a doctor with the specified specialization who is available and not booked
        available_doctor = None
        for doctor in appointments:
            if doctor["Specialist"] == user_special and doctor["drstatus"] == "Available" and doctor["booking_status"] == "Not Booked":
                available_doctor = doctor
                break

        if available_doctor:
            drid = available_doctor["drid"]
            # Simulate booking logic by updating the hardcoded list (in memory)
            available_doctor["booking_status"] = "Booked"
            msg = 'Your appointment has been booked with {}! We will send the meet link through email.'.format(available_doctor["drname"])
        else:
            msg = 'Now, the doctor is not available! You can meet a General Doctor for instant relief!'

    return render_template('response.html', msg=msg)


# @app.route('/appointment', methods=['GET'])
# def appointment():
#     cursor.execute("SELECT * from appointment")
#     result = cursor.fetchall()
#     return render_template('appointment.html', info_table=result)

# @app.route('/appoint', methods=['GET','POST'])
# def appoint():
#     msg=''
#     if request.method == 'POST':
#         full_name = request.form['name']
#         user_email=request.form['email']
#         user_phone=request.form['phone']
#         user_date=request.form['date']
#         user_time=request.form['time']
#         user_special=request.form['specialization']
#         sql1 = "SELECT * FROM appointment WHERE Specialist = %s AND drstatus = 'Available' AND booking_status='Not Booked'"
#         val1 = (user_special,)
#         cursor.execute(sql1,val1)
#         result1=cursor.fetchone()
#         if result1:
#             drid=result1[0]
#             sql = "INSERT INTO patient (name, email,phone_no,appoint_date,appoint_time,drid) VALUES (%s, %s, %s, %s, %s, %s)"
#             val = (full_name, user_email, user_phone, user_date, user_time, drid)
#             cursor.execute(sql, val)
#             sql2 = "UPDATE appointment SET booking_status = 'Booked' WHERE drid = %s"
#             val2=(drid,)
#             cursor.execute(sql2,val2)
#             connection.commit()
#             msg='Your appointment has been booked!!!! Will send the meet link through mail!!!'
#         else:
#             msg='Now, the doctor is not available!!!! Can meet General Doctor for instant relief!!!'
#     return render_template('response.html', msg=msg)

@app.route('/chat')
def chat():
    return render_template('chat.html')

def tokenize(text):
  return [stemmer.stem(token) for token in word_tokenize(text)]

english_stopwords= stopwords.words('english')

def vectorizer():
    vectorizer= TfidfVectorizer(tokenizer=tokenize, stop_words=english_stopwords, )
    return vectorizer

class RNN_model(nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn= nn.RNN(input_size=1080, hidden_size=240,num_layers=1, nonlinearity= 'relu', bias= True)
    self.output= nn.Linear(in_features=240, out_features=24)

  def forward(self, x):
    y, hidden= self.rnn(x)
    x= self.output(y)
    return(x)

df= pd.read_csv('Symptom2Disease.csv')
df.drop('Unnamed: 0', axis= 1, inplace= True)

df.drop_duplicates(inplace= True)
train_data, test_data= train_test_split(df, test_size=0.15, random_state=42 )

model = RNN_model()
model.load_state_dict(torch.load('pretrained_symtom_to_disease_model.pth', map_location=torch.device('cpu')))
vectorizer = vectorizer()
vectorizer.fit(train_data.text)

class_names= {0: 'Acne',
              1: 'Arthritis',
              2: 'Bronchial Asthma',
              3: 'Cervical spondylosis',
              4: 'Chicken pox',
              5: 'Common Cold',
              6: 'Dengue',
              7: 'Dimorphic Hemorrhoids',
              8: 'Fungal infection',
              9: 'Hypertension',
              10: 'Impetigo',
              11: 'Jaundice',
              12: 'Malaria',
              13: 'Migraine',
              14: 'Pneumonia',
              15: 'Psoriasis',
              16: 'Typhoid',
              17: 'Varicose Veins',
              18: 'allergy',
              19: 'diabetes',
              20: 'drug reaction',
              21: 'gastroesophageal reflux disease',
              22: 'peptic ulcer disease',
              23: 'urinary tract infection'
              }
disease_advice = {
    'Acne': "Maintain a proper skincare routine, avoid excessive touching of the affected areas, and consider using over-the-counter topical treatments. If severe, consult a dermatologist.",
    'Arthritis': "Stay active with gentle exercises, manage weight, and consider pain-relief strategies like hot/cold therapy. Consult a rheumatologist for tailored guidance.",
    'Bronchial Asthma': "Follow prescribed inhaler and medication regimen, avoid triggers like smoke and allergens, and have an asthma action plan. Regular check-ups with a pulmonologist are important.",
    'Cervical spondylosis': "Maintain good posture, do neck exercises, and use ergonomic support. Physical therapy and pain management techniques might be helpful.",
    'Chicken pox': "Rest, maintain hygiene, and avoid scratching. Consult a doctor for appropriate antiviral treatment.",
    'Common Cold': "Get plenty of rest, stay hydrated, and consider over-the-counter remedies for symptom relief. Seek medical attention if symptoms worsen or last long.",
    'Dengue': "Stay hydrated, rest, and manage fever with acetaminophen. Seek medical care promptly, as dengue can escalate quickly.",
    'Dimorphic Hemorrhoids': "Follow a high-fiber diet, maintain good hygiene, and consider stool softeners. Consult a doctor if symptoms persist.",
    'Fungal infection': "Keep the affected area clean and dry, use antifungal creams, and avoid sharing personal items. Consult a dermatologist if it persists.",
    'Hypertension': "Follow a balanced diet, exercise regularly, reduce salt intake, and take prescribed medications. Regular check-ups with a healthcare provider are important.",
    'Impetigo': "Keep the affected area clean, use prescribed antibiotics, and avoid close contact. Consult a doctor for proper treatment.",
    'Jaundice': "Get plenty of rest, maintain hydration, and follow a doctor's advice for diet and medications. Regular monitoring is important.",
    'Malaria': "Take prescribed antimalarial medications, rest, and manage fever. Seek medical attention for severe cases.",
    'Migraine': "Identify triggers, manage stress, and consider pain-relief medications. Consult a neurologist for personalized management.",
    'Pneumonia': "Follow prescribed antibiotics, rest, stay hydrated, and monitor symptoms. Seek immediate medical attention for severe cases.",
    'Psoriasis': "Moisturize, use prescribed creams, and avoid triggers. Consult a dermatologist for effective management.",
    'Typhoid': "Take prescribed antibiotics, rest, and stay hydrated. Dietary precautions are important. Consult a doctor for proper treatment.",
    'Varicose Veins': "Elevate legs, exercise regularly, and wear compression stockings. Consult a vascular specialist for evaluation and treatment options.",
    'allergy': "Identify triggers, manage exposure, and consider antihistamines. Consult an allergist for comprehensive management.",
    'diabetes': "Follow a balanced diet, exercise, monitor blood sugar levels, and take prescribed medications. Regular visits to an endocrinologist are essential.",
    'drug reaction': "Discontinue the suspected medication, seek medical attention if symptoms are severe, and inform healthcare providers about the reaction.",
    'gastroesophageal reflux disease': "Follow dietary changes, avoid large meals, and consider medications. Consult a doctor for personalized management.",
    'peptic ulcer disease': "Avoid spicy and acidic foods, take prescribed medications, and manage stress. Consult a gastroenterologist for guidance.",
    'urinary tract infection': "Stay hydrated, take prescribed antibiotics, and maintain good hygiene. Consult a doctor for appropriate treatment."
}

def predict_disease(message):
    transform_text = vectorizer.transform([message])
    transform_text = torch.tensor(transform_text.toarray()).to(torch.float32)
    model.eval()
    with torch.inference_mode():
        y_logits = model(transform_text)
        pred_prob = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
    return class_names[pred_prob.item()]

@app.route('/prediction', methods=['POST'])
def prediction():
    data = request.get_json()
    symptoms = data.get('symptoms')
    disease = predict_disease(symptoms) 
    advice = disease_advice.get(disease, "Consult a doctor for more information.")
    return jsonify({"disease": disease, "advice": advice})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

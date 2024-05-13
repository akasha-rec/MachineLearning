from flask import Flask, render_template, request
from konlpy.tag import Okt

import joblib
import re

app = Flask(__name__) #app.py를 가리켜
app.debug = True #플라스크 관련 설정

okt = Okt()

model_lr = None
tfidf_vector = None

model_nb = None
dtm_vector = None


def load_lr():
    global model_lr, tfidf_vector
    model_lr = joblib.load("model/movie_lr.pkl")
    tfidf_vector = joblib.load("model/movie_lr_dtm.pkl")

def load_nb():
    global model_nb, dtm_vector
    model_nb = joblib.load("model/movie_nb.pkl")
    dtm_vector = joblib.load("model/movie_nb_dtm.pkl")

def tw_tokenizer(text):
    token_ko = okt.morphs(text)
    return token_ko


def lt_t(text): #전처리
    review = re.sub(r"\d+", " ", text)
    text_vector = tfidf_vector.transform([review]) #callable
    return text_vector

def lt_nb(text):
    stopwords = ["은", "는", "이", "가"]
    review = text.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]", "")
    morphs = okt.morphs(review, stem=True) #토큰 분리
    test = " ".join(morph for morph in morphs if not morph in stopwords)
    test_dtm = dtm_vector.transform(test)
    return test_dtm

@app.route("/")
def index():
    menu = {
        "home" : True,
        "senti" : False
    }
    return render_template("home.html", menu=menu) #관례 > templates 폴더명 바꾸면 못 찾음

@app.route("/senti", methods=["GET", "POST"])
def senti():
    if request.method == "GET":
        menu = {
            "home" : False,
            "senti" : True
        }
        return render_template("senti.html", menu=menu)
    else: #역직렬화 > 입력한 감상평에 대한 결과값(1/0)
        review = request.form["review"]
        review_text = lt_t(review)
        lr_result = model_lr.predict(review_text)[0]
        review_text2 = lt_nb(review)
        nb_result = model_lr.predict(review_text2)[0]
        lr = "긍정" if lr_result else "부정"
        nb = "긍정" if nb_result else "부정"
        movie = {"review" : review, "lr" : lr, "nb" : nb}
        return render_template("senti_result.html", menu=menu, movie=movie)

if __name__ == "__main__":
    load_lr()
    load_nb()
    app.run()
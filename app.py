from flask import Flask, render_template, request
from src.pipeline.predict import ModelPredictor

app = Flask(__name__) 
predictor = ModelPredictor()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        tweet = request.form["tweet"]
        sentiment = predictor.predict_sentiment(tweet)
        return render_template("index.html", tweet=tweet, sentiment=sentiment)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)


#

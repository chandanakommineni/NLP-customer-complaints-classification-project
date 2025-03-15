from flask import Flask, render_template, request, redirect, url_for
import joblib

app = Flask(__name__)

# Load trained model
vectorizer, model = joblib.load("bank_complaint_classifier.pkl")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            content = file.read().decode("utf-8")
            complaints = content.strip().split("\n")
            results = {}

            for complaint in complaints:
                if complaint.strip():
                    X_test = vectorizer.transform([complaint])
                    category = model.predict(X_test)[0]
                    results[complaint] = category

            return render_template("results.html", results=results)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)

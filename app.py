from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Space Debris Detection Running 🚀"

def handler(request):
    return app
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/api")
def api():
    return jsonify({"message": "Backend working 🚀"})

def handler(request):
    return app
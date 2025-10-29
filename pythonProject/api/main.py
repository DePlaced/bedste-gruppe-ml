from flask import Flask, request, jsonify

app = Flask(__name__)

@app.get("/")
def health():
    return jsonify(status="ok")

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    if "features" in data:
        pred = float(sum(data["features"]))  # demo numeric logic
    else:
        text = str(data.get("sample", ""))
        pred = len(text)                     # demo text logic
    return jsonify(prediction=pred, received=data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)





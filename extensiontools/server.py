from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get("text", "")

    response = {
        "summary": text[:200],  # Example: Just returning first 200 characters
        "word_count": len(text.split())
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000)

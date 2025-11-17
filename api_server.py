# api_server.py
from flask import Flask, request, jsonify
from sentiment_analyzer import SentimentAnalyzer
import json

app = Flask(__name__)
analyzer = SentimentAnalyzer()

@app.route('/')
def home():
    return """
    <h1>🎭 Sentiment Analysis API</h1>
    <h2>Available Endpoints:</h2>
    <ul>
        <li><b>POST /analyze</b> - Analyze sentiment of text</li>
        <li><b>GET /test</b> - Test with sample text</li>
        <li><b>GET /stats</b> - Get API statistics</li>
    </ul>
    <h3>Example Usage:</h3>
    <pre>
    POST /analyze
    Content-Type: application/json

    {"text": "This product is amazing!"}
    </pre>
    """

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']

        simple = analyzer.analyze_sentiment_simple(text)
        advanced = analyzer.analyze_sentiment_advanced(text)

        response = {
            'text': text,
            'simple_analysis': simple,
            'advanced_analysis': advanced,
            'recommendation': 'This review appears to be ' + advanced['sentiment'].lower()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    test_reviews = [
        "This product is absolutely fantastic!",
        "Terrible quality, very disappointed.",
        "It's okay, nothing special really."
    ]

    results = []
    for review in test_reviews:
        advanced = analyzer.analyze_sentiment_advanced(review)
        results.append({
            'text': review,
            'sentiment': advanced['sentiment'],
            'confidence': f"{advanced['confidence']:.1f}%"
        })

    return jsonify({
        'test_results': results,
        'message': 'API is working correctly!'
    })

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({
        'api_version': '1.0',
        'endpoints': 3,
        'supported_languages': ['English'],
        'methods': ['Simple word counting', 'Advanced ML-based'],
        'creator': 'CSCI 130 Student'
    })

if __name__ == '__main__':
    print("🚀 Starting Sentiment Analysis API...")
    print("📍 Visit http://localhost:5000")
    app.run(debug=True, port=5000)
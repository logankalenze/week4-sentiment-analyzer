# Sentiment Analysis System
## Overview
This project implements sentiment analysis for product reviews using Natural Language
Processing (NLP) techniques from Chapter 23.
## Features
- **Two Analysis Methods:**
 - Simple: Word counting based on positive/negative word lists
 - Advanced: Machine learning using TextBlob
- **REST API** for easy integration
- **Visualization** of sentiment distributions
- **Text preprocessing** including tokenization and stop word removal
## How to Run
### 1. Install Requirements
```bash
pip install -r requirements.txt
```
### 2. Test the Analyzer
```bash
python test_sentiment.py
```
### 3. Start the API Server
```bash
python api_server.py
```
Then visit http://localhost:5000
## API Usage Example
```python
import requests
response = requests.post('http://localhost:5000/analyze',
 json={'text': 'This product is amazing!'})
print(response.json())
```
## Understanding NLP Concepts
### Tokenization
Breaking text into words: "I love this" → ["I", "love", "this"]
### Stop Words
Common words we filter out: "the", "is", "at", "which"
### Sentiment Analysis
Determining if text is positive, negative, or neutral
## Challenges Encountered
One challenge I ran into was making sure the text preprocessing didn’t accidentally remove words that actually mattered for detecting sentiment. I also struggled a bit with understanding how TextBlob’s polarity scores translated into real positive or negative results because the numbers felt abstract at first. Another issue was getting the Flask routes to return clean JSON without errors since even small formatting mistakes caused problems.
## Real-World Applications
Sentiment analysis like this can be used for monitoring customer reviews so companies can see how people feel about their products. It can also help support teams catch negative feedback faster and respond to issues before they get worse. Another common use is analyzing social media posts, which helps businesses understand trends or public reactions to new releases.
## What I Learned
I learned how simple rule-based methods and ML-based methods can give different results and why combining them can be useful. I also got a better understanding of how text preprocessing affects accuracy and why things like removing stop words actually matter. Building the API helped me see how projects can be turned into real tools that other programs or users can interact with. Overall, this project gave me a clearer idea of how NLP works in practice rather than just in theory.
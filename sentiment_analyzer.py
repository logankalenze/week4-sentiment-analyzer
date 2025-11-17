# sentiment_analyzer.py
from textblob import TextBlob
import pandas as pd
import re
from collections import Counter
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

class SentimentAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.positive_words = {
            'good','great','excellent','amazing','wonderful','fantastic',
            'love','best','perfect','happy'
        }
        self.negative_words = {
            'bad','terrible','awful','horrible','worst','hate',
            'disappointing','useless','waste','poor'
        }

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        return words

    def analyze_sentiment_simple(self, text):
        words = self.preprocess_text(text)

        positive_count = sum(1 for w in words if w in self.positive_words)
        negative_count = sum(1 for w in words if w in self.negative_words)

        if positive_count > negative_count:
            sentiment = "Positive"
            confidence = (positive_count / (positive_count + negative_count + 1)) * 100
        elif negative_count > positive_count:
            sentiment = "Negative"
            confidence = (negative_count / (positive_count + negative_count + 1)) * 100
        else:
            sentiment = "Neutral"
            confidence = 50

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_words_found': positive_count,
            'negative_words_found': negative_count
        }

    def analyze_sentiment_advanced(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        confidence = abs(polarity) * 100

        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'confidence': min(confidence, 100),
            'subjectivity': blob.sentiment.subjectivity
        }

    def analyze_multiple_reviews(self, reviews):
        results = []
        for review in reviews:
            simple = self.analyze_sentiment_simple(review)
            advanced = self.analyze_sentiment_advanced(review)

            results.append({
                'review': review[:50] + '...' if len(review) > 50 else review,
                'simple_sentiment': simple['sentiment'],
                'advanced_sentiment': advanced['sentiment'],
                'polarity': advanced['polarity']
            })

        df = pd.DataFrame(results)

        print("\n📊 SENTIMENT ANALYSIS RESULTS")
        print("=" * 50)
        print("\nSimple Method Results:")
        print(df['simple_sentiment'].value_counts())
        print("\nAdvanced Method Results:")
        print(df['advanced_sentiment'].value_counts())

        return df

    def visualize_sentiment(self, reviews):
        polarities = []
        sentiments = []

        for review in reviews:
            result = self.analyze_sentiment_advanced(review)
            polarities.append(result['polarity'])
            sentiments.append(result['sentiment'])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sentiment_counts = Counter(sentiments)
        axes[0].bar(sentiment_counts.keys(), sentiment_counts.values(),
                    color=['green','gray','red'])
        axes[0].set_title('Sentiment Distribution')
        axes[0].set_ylabel('Number of Reviews')

        axes[1].hist(polarities, bins=20, edgecolor='black')
        axes[1].set_title('Polarity Score Distribution')
        axes[1].set_xlabel('Polarity (-1 = Negative, +1 = Positive)')
        axes[1].set_ylabel('Count')
        axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('sentiment_visualization.png')
        print("\n📈 Saved visualization as 'sentiment_visualization.png'")

        return fig
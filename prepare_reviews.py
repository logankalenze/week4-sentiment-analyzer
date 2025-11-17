# prepare_reviews.py
import pandas as pd
import random

def create_review_dataset():
    reviews = [
        {"text": "This product is amazing! Best purchase ever!", "sentiment": 1},
        {"text": "Excellent quality and fast shipping. Very happy!", "sentiment": 1},
        {"text": "Love it! Works perfectly and looks great.", "sentiment": 1},
        {"text": "Exceeded my expectations. Highly recommend!", "sentiment": 1},
        {"text": "Five stars! Fantastic product and great price.", "sentiment": 1},
        {"text": "So glad I bought this. Life changing!", "sentiment": 1},
        {"text": "Perfect! Exactly what I needed. Thank you!", "sentiment": 1},
        {"text": "Outstanding quality. Will buy again for sure.", "sentiment": 1},
        {"text": "Best product in its category. Worth every penny!", "sentiment": 1},
        {"text": "Incredible! My whole family loves it.", "sentiment": 1},

        {"text": "Terrible product. Complete waste of money.", "sentiment": 0},
        {"text": "Broke after one day. Very disappointed.", "sentiment": 0},
        {"text": "Horrible quality. Returning immediately.", "sentiment": 0},
        {"text": "Does not work as advertised. Avoid!", "sentiment": 0},
        {"text": "Worst purchase ever. Total garbage.", "sentiment": 0},
        {"text": "Cheap and poorly made. Not worth it.", "sentiment": 0},
        {"text": "Completely useless. I want my money back.", "sentiment": 0},
        {"text": "Awful experience. Customer service was rude too.", "sentiment": 0},
        {"text": "Defective product. Very frustrating!", "sentiment": 0},
        {"text": "Don't buy this. You'll regret it.", "sentiment": 0},

        {"text": "It's okay. Not great but not terrible either.", "sentiment": 1},
        {"text": "Product works but shipping took forever.", "sentiment": 0},
        {"text": "Good product but overpriced for what you get.", "sentiment": 1},
        {"text": "Has potential but needs improvement.", "sentiment": 0}
    ]

    df = pd.DataFrame(reviews)
    return df

if __name__ == "__main__":
    data = create_review_dataset()
    data.to_csv('data/training_reviews.csv', index=False)
    print(f"Created {len(data)} sample reviews")
import json

with open('review.json', 'rb') as f, open('raw_review.txt', 'w', encoding='UTF8') as w:
    for line in f:
        review = json.loads(line)
        w.write(review['text'])

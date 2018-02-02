import json

learning_data_root = 'data/learning/'

with open(learning_data_root+'review.json', 'rb') as f, open(learning_data_root+'raw_review.txt', 'w', encoding='UTF8') as w:
    for line in f:
        review = json.loads(line)
        w.write(review['text'])

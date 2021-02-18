import utilities
import json
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from distutils.util import strtobool

# def predict(model, tfidf_fit, input_term, label):
#     predictions = []
#     probas = model.predict_proba(tfidf_fit.transform([input_term]).tolist()

#     [predictions.append([x[1] for x in [sorted(zip(example, model.classes_), reverse=True)][0]]) for example in probas]
    
#     for idx, example in enumerate(predictions):
#         data = {"term": input_term, "label": label, "predicted_labels": example}
    
#     return data

def main():
    output_path = '../data/outputs/experiment_3c.'

    terms_path = '../data/terms/train.json'
    test_set_path = '../data/terms/test_set.json'

    #parameters
    max_df = 1.0
    min_gram = 1
    max_gram = 1
    # data loading 
    terms_df = pd.DataFrame(json.load(open(terms_path)))
    test = json.load(open(test_set_path))

    # path to training/testing data
    train_set_path = '../data/terms/train_set.json'
    test_set_path = '../data/terms/test_set.json'
    valid_set_path = '../data/terms/valid_set.json'

    # if separated data does not exist, create them
    if not os.path.isfile(train_set_path):
        terms_path = '../data/terms/train.json'
        term_list = json.load(open(terms_path))

        train, valid, test = utilities.train_test_split(term_list, 3)

        with open('../data/terms/train_set.json', 'w+') as outfile:
            json.dump(train, outfile, indent=2)

        with open('../data/terms/test_set.json', 'w+') as outfile:
            json.dump(test, outfile, indent=2)
        
        with open('../data/terms/valid_set.json', 'w+') as outfile:
            json.dump(valid, outfile, indent=2)
    
    #data loading
    train_df = pd.DataFrame(json.load(open(train_set_path)))
    test_df = pd.DataFrame(json.load(open(test_set_path)))
    valid_df = pd.DataFrame(json.load(open(valid_set_path)))

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(min_gram,max_gram), max_df=max_df, lowercase=False)

    tfidf_fit = tfidf_vectorizer.fit(train_df['term'])
    tfidf_train = tfidf_fit.transform(train_df['term'])
    
    labels = train_df['label'].tolist()
    
    # X_train, _, Y_train, _ = train_test_split(tfidf_train, labels, test_size=0.00001, shuffle=False, random_state=9)

    model = RandomForestClassifier(random_state=None)

    model = utilities.train(tfidf_train, labels, model)

    predictions = []
    X_test = tfidf_fit.transform(test_df['term'])
    
    probas = model.predict_proba(X_test).tolist()

    [predictions.append([x[1] for x in [sorted(zip(example, model.classes_), reverse=True)][0]]) for example in probas]
    
    data = []
    output = output_path + str(min_gram) + '.' + str(max_gram) + '.' + str(int(max_df*100))+'.json'
    for idx, example in enumerate(predictions):
        data.append({"term": test[idx]["term"], "label": test[idx]["label"], "predicted_labels": example})
    json.dump(data, open(output, "w"), indent=4)

if __name__ == '__main__':
    main()
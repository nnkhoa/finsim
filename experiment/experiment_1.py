import utilities
import json
import os 
from sklearn.linear_model import LogisticRegression

from distutils.util import strtobool

from gensim.scripts.glove2word2vec import glove2word2vec

# Experiment the baseline 2 with custom embedding(from baseline 2), Google News Word2Vec, Stanford 6B Glove, and Wiki News FastText embedding

def main():
    # load training data and embedding
    
    embedding_list = ['custom_w2v_300d.txt', 'GoogleNews-vectors-negative300.bin.gz', 'glove.6B.300d.txt', 'wiki-news-300d-1M.vec']
    terms_path = '../data/terms/train.json'
    output_path = '../data/outputs/experiment_1.'

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
    train = json.load(open(train_set_path))
    test = json.load(open(test_set_path))
    valid = json.load(open(valid_set_path))

    for embedding in embedding_list:
        # load embedding
        binary = False

        if 'bin.gz' in embedding:
            binary = True

        embedding_path = '../models/' + embedding
        
        if 'glove' in embedding:
            glove2word2vec(embedding_path, '../models/w2v.' + embedding)
            embedding_path = '../models/w2v.' + embedding    

        embeddings = utilities.loadWord2Vec(embedding_path, binary)
        
        model = LogisticRegression(solver='lbfgs', max_iter=1000)

        X_train, Y_train = utilities.get_X_Y(embeddings, train)
        model = utilities.train(X_train, Y_train, model)

        # test
        predictions = []
        X_test, Y_test = utilities.get_X_Y(embeddings, test)
        probas = model.predict_proba(X_test).tolist()

        [predictions.append([x[1] for x in [sorted(zip(example, model.classes_), reverse=True)][0]]) for example in probas]
        
        # write result
        output = output_path + embedding + '.json'
        data = []
        for idx, example in enumerate(predictions):
            data.append({"term": test[idx]["term"], "label": test[idx]["label"], "predicted_labels": example})
        json.dump(data, open(output, "w"), indent=4)


if __name__ == '__main__':
    main()
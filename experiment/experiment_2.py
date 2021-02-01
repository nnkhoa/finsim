import utilities
import json

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics

from distutils.util import strtobool

from gensim.scripts.glove2word2vec import glove2word2vec

# Experiment the SVM with custom embedding(from baseline 2), Google News Word2Vec, Stanford 6B Glove, and Wiki News FastText embedding

def main():
    # load training data and embedding
    
    embedding_list = ['custom_w2v_300d.txt', 'GoogleNews-vectors-negative300.bin.gz', 'glove.6B.300d.txt', 'wiki-news-300d-1M.vec']
    terms_path = '../data/terms/train.json'
    output_path = '../data/outputs/experiment_2.'

    term_list = json.load(open(terms_path))

    # split
    train, valid, test = utilities.train_test_split(term_list, 3)

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
        
        # model = SVC(probability=True)
        model = LinearSVC()
        X_train, Y_train = utilities.get_X_Y(embeddings, train)
        model = utilities.train(X_train, Y_train, model)

        # test
        predictions = []
        X_test, Y_test = utilities.get_X_Y(embeddings, test)
        Y_pred = model.predict(X_test)
        
        # probas = model.predict_proba(X_test).tolist()

        # [predictions.append([x[1] for x in [sorted(zip(example, model.classes_), reverse=True)][0]]) for example in probas]
        
        # write result
        # output = output_path + embedding + '.json'
        # data = []
        # for idx, example in enumerate(predictions):
        #     data.append({"term": test[idx]["term"], "label": test[idx]["label"], "predicted_labels": example})
        # json.dump(data, open(output, "w"), indent=4)

        # write result
        output = output_path + embedding + '.xlsx'
        # data = []
        # for x_test, y_test, y_pred in zip(test, Y_test, Y_pred.tolist()):
        #     data.append({"term": x_test['term'], "label": y_test, "predicted_labels": [y_pred]})
        # json.dump(data, open(output, "w"), indent=4)
        
        report = metrics.classification_report(Y_test, Y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_excel(output)


if __name__ == '__main__':
    main()
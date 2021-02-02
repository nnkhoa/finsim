import utilities
import json
import os

if __name__ == '__main__':
    labels_path = '../data/tagset/finsim.json'
    train_set_path = '../data/terms/train_set.json'
    test_set_path = '../data/terms/test_set.json'
    valid_set_path = '../data/terms/valid_set.json'

    label_list = json.load(open(labels_path))
    
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
    
    train = json.load(open(train_set_path))
    test = json.load(open(test_set_path))
    valid = json.load(open(valid_set_path))

    utilities.convert_data(train, label_list)
import utilities

import json
import os
import math
import logging

from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

if __name__ == '__main__':
    labels_path = '../data/tagset/finsim.json'
    train_set_path = '../data/terms/train_set.json'
    test_set_path = '../data/terms/test_set.json'
    valid_set_path = '../data/terms/valid_set.json'
    save_trained_model_path = '../models/experiment4.'

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

    train_data = utilities.convert_data(train, label_list)
    test_data = utilities.convert_data(test, label_list)
    valid_data = utilities.convert_data(valid, label_list)

    train_samples = [InputExample(texts=[entry[0], entry[1]], label=float(entry[2])) for entry in train_data]
    test_samples = [InputExample(texts=[entry[0], entry[1]], label=float(entry[2])) for entry in test_data]
    valid_samples = [InputExample(texts=[entry[0], entry[1]], label=float(entry[2])) for entry in valid_data]

    # bert_models = ['distilbert-base-cased', 
                    # 'distilbert-base-cased-finetuned-sst-2-english', 
                    # 'textattack/distilbert-base-cased-SST-2',
                    # 'bert-base-cased',
                    # 'bert-large-cased']
    
    # bert_model = 'textattack/distilbert-base-uncased-SST-2'
    bert_model = 'bert-large-uncased'

    epochs = [50, 100, 150]

    for num_epochs in epochs:
        print(bert_model)

        word_embedding_model = models.Transformer(bert_model)

        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                        pooling_mode_mean_tokens=True,
                                        pooling_mode_cls_token=False,
                                        pooling_mode_max_tokens=False)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        train_batch_size = 16
        # num_epochs = 20

        train_data_loader = DataLoader(train_samples, shuffle=False, batch_size=train_batch_size)
        train_loss = losses.CosineSimilarityLoss(model=model)

        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(valid_samples, name='valid')

        # warm-up step (skip? guess not cuz skip = worse performance)
        warmup_steps = math.ceil(len(train_samples) * num_epochs + 0.1)

        model.fit(train_objectives=[(train_data_loader, train_loss)],
                    evaluator=evaluator,
                    epochs=num_epochs,
                    warmup_steps=warmup_steps,
                    evaluation_steps=1000,
                    output_path=save_trained_model_path + bert_model.replace('/','.') + '.' + str(num_epochs))
        
        # test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='test')
        # test_evaluator(model)
        
        test_output = []

        for i in range(0, len(test_data), 10):
            term = [test_data[i][0]]
            labels = []
            groundtruth = [entry['label'] for entry in test if entry['term'] == term[0]]

            for j in range(0, 10):
                labels.append(test_data[i+j][1])
            
            term_embeddings = model.encode(term, convert_to_tensor=True, show_progress_bar=False)
            labels_embeddings = model.encode(labels, convert_to_tensor=True, show_progress_bar=False)

            cosine_score = util.pytorch_cos_sim(term_embeddings, labels_embeddings)
            
            distance = []
            
            for j in range(len(labels)):
                distance.append((labels[j], cosine_score[0][j]))

            distance.sort(key=lambda x: x[1], reverse=True)
            ranking = [label[0] for label in distance]

            entry = {"term": term[0], "label": groundtruth[0], "predicted_labels": ranking}

            test_output.append(entry)
        
        output = '../data/outputs/experiment_4.'+bert_model.replace('/','.')+'.'+str(num_epochs)+'.json'
        json.dump(test_output, open(output, "w"), indent=4)
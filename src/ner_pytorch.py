import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
from enum import Enum
import time
import re
import gensim
import operator

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from torch.autograd import Variable

def get_ner_tag_switcher():
    # PADDING SHOULD ALWAYS BE THE LAST CLASS

    # indices = {
    #     'O': 0,
    #     'I-ORG': 1,
    #     'B-ORG': 2,
    #     'I-PER': 3,
    #     'B-PER': 4,
    #     'I-LOC': 5,
    #     'B-LOC': 6,
    #     'I-MISC': 7,
    #     'B-MISC': 8,
    #     '<PAD>': 9,
    # }

    indices = {
        'O': 0,
        'I-ORG': 1,
        'I-PER': 2,
        'I-LOC': 3,
        'I-MISC': 4,
        '<PAD>': 5,
    }

    return indices

def get_reverse_ner_tag_switcher():
    return {v: k for k, v in get_ner_tag_switcher().items()}

class Optimizer(Enum):
    sgd = 'sgd'
    adam = 'adam'
    adagrad = 'adagrad'
    adadelta = 'adadelta'
    rmsprop = 'rmsprop'
    nsgd = 'nsgd'

    def __str__(self):
        return self.value

class NER_LSTM_BiDi(torch.nn.Module):

    def __init__(self, vector_dimension=300, output_size=9, batch_size=256, padding_token='<PAD>', device=torch.device('cpu')):
        super(NER_LSTM_BiDi, self).__init__()

        self.input_size = vector_dimension
        self.hidden_size = 512
        self.output_size = output_size
        self.num_layers = 2
        self.padding_token = padding_token
        self.dropout_probability = 0.2
        self.device = device

        self.lstm = torch.nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=self.dropout_probability)

        self.fc0 = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.relu0 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=self.dropout_probability)
        self.fc1 = torch.nn.Linear(self.hidden_size, 64)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, self.output_size)
        # self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=0.5)
        # self.dropout3 = torch.nn.Dropout(p=0.5)

        self.fc_in = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu_in = torch.nn.ReLU()

    def forward(self, x, sequence_lengths):
        # (minibatch_size, seq_length, features)

        x = self.fc_in(x)

        x = self.relu_in(x)

        residual = x

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        x = torch.nn.utils.rnn.pack_padded_sequence(x, sequence_lengths, batch_first=True)

        x, _ = self.lstm(x)

        # undo the packing operation
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.fc0(x)

        x = self.relu0(x)

        x += residual

        x = self.dropout1(x)

        x = self.fc1(x)

        x = self.relu1(x)

        x = self.dropout2(x)

        x = self.fc2(x)

        # x = self.relu2(x)

        # x = self.dropout3(x)

        # No explicit softmax if we are using PyTorch's inbuilt CrossEntropyLoss which is log softmax + NLL loss. 
        # x = self.softmax(x)
        return x

class CONLL_2003_Dataset(Dataset):
    """CoNLL 2003 dataset."""

    def __init__(self, data_root_dir, word_embeddings, transform=None, device=torch.device('cpu')):
        """
        Args:
            data_root_dir (string): Directory with all the sentences.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_root_dir = data_root_dir
        self.transform = transform
        with open(data_root_dir + '/__META__', 'r') as f:
            head = [next(f) for x in range(2)]
        self.numSentences = int(head[0][:-1])
        self.filename_prefix = head[1][:-1]
        self.device = device
        self.word_embeddings = word_embeddings
        self.ner_tag_switcher = get_ner_tag_switcher()

    def __len__(self):
        return self.numSentences

    def __getitem__(self, idx):
        sentence_file_name = os.path.join(self.data_root_dir,
                                self.filename_prefix + str(idx))
        tokens = []
        tokens_vectors = []
        chunk_tags = []
        pos_tags = []
        ner_tags = []
        with open(sentence_file_name, 'r') as f:
            for line in f:
                token_details = re.split(r'\s+', line)
                this_token = token_details[0].strip()
                if this_token not in self.word_embeddings.vocab:
                    self.word_embeddings[this_token] = np.random.uniform(-0.25, 0.25, 300).astype('f')

                word_vector = torch.tensor(self.word_embeddings[this_token], dtype=torch.float32, requires_grad=False)
                tokens.append(this_token)
                tokens_vectors.append(word_vector)
                chunk_tags.append(token_details[1])
                pos_tags.append(token_details[2])

                ner_tag = token_details[3]
                if ner_tag.startswith('B-'):
                    ner_tag = 'I-' + ner_tag[2:]

                ner_tags.append(self.ner_tag_switcher[ner_tag])

        sample = (
            tokens,
            tokens_vectors,
            chunk_tags,
            pos_tags,
            ner_tags
        )
        if self.transform:
            sample = self.transform(sample)

        return sample

class AIDA_YAGO_Dataset(Dataset):
    """AIDA YAGO dataset."""

    def __init__(self, data_root_dir, word_embeddings, transform=None, device=torch.device('cpu')):
        """
        Args:
            data_root_dir (string): Directory with all the sentences.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_root_dir = data_root_dir
        self.transform = transform
        with open(data_root_dir + '/__META__', 'r') as f:
            head = [next(f) for x in range(2)]
        self.numSentences = int(head[0][:-1])
        self.filename_prefix = head[1][:-1]
        self.device = device
        self.word_embeddings = word_embeddings
        self.ner_tag_switcher = get_ner_tag_switcher()

    def __len__(self):
        return self.numSentences

    def __getitem__(self, idx):
        sentence_file_name = os.path.join(self.data_root_dir,
                                self.filename_prefix + str(idx))
        tokens = []
        tokens_vectors = []
        wikipedia_links = []
        complete_entities = []
        with open(sentence_file_name, 'r') as f:
            for line in f:
                token_details = line.split('\t')
                this_token = token_details[0].strip()
                if this_token not in self.word_embeddings.vocab:
                    self.word_embeddings[this_token] = np.random.uniform(-0.25, 0.25, 300).astype('f')

                wikipedia_link = ''
                if len(token_details) >= 5 and 'en.wikipedia.org' in line:
                    wikipedia_link = token_details[4]
                
                complete_entity = this_token
                if len(token_details) >= 4:
                    complete_entity = token_details[3]

                word_vector = torch.tensor(self.word_embeddings[this_token], dtype=torch.float32, requires_grad=False)
                tokens.append(this_token)
                tokens_vectors.append(word_vector)
                wikipedia_links.append(wikipedia_link)
                complete_entities.append(complete_entity)

        sample = (
            tokens,
            tokens_vectors,
            wikipedia_links,
            complete_entities
        )
        if self.transform:
            sample = self.transform(sample)

        return sample

class NumpyWordVector(dict):
   def __init__(self,*arg,**kw):
        super(NumpyWordVector, self).__init__(*arg, **kw)
        self.vector_size = len(next(iter(self.values())))
        self.vocab = self.keys()

def get_optimizer(optimizer, network):
    LEARNING_RATE = 0.0005
    MOMENTUM = 0.9
    switcher = {
        Optimizer.sgd: torch.optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM),
        Optimizer.adam: torch.optim.Adam(network.parameters(), lr=LEARNING_RATE),
        Optimizer.adagrad: torch.optim.Adagrad(network.parameters(), lr=LEARNING_RATE),
        Optimizer.adadelta: torch.optim.Adadelta(network.parameters(), lr=LEARNING_RATE),
        Optimizer.rmsprop: torch.optim.RMSprop(network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM),
        Optimizer.nsgd: torch.optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True),
    }
    opt = switcher.get(optimizer, switcher[Optimizer.adam])
    return opt

def get_collater(word_embeddings, word_embeddings_dimension, pad_first=False, padding_token='<PAD>'):
    padding_token_embedding = np.random.uniform(-0.25, 0.25, word_embeddings_dimension).astype('f')
    if padding_token not in word_embeddings.vocab:
        word_embeddings[padding_token] = padding_token_embedding
    
    padding_token_embedding = torch.tensor(padding_token_embedding, dtype=torch.float32, requires_grad=False)
    padding_ner_tag = get_ner_tag_switcher()[padding_token]

    def collate_to_minibatch(data):
        """Creates mini-batch tensors from the list of tuples (image, caption).
        
        We should build custom collate_fn rather than using default collate_fn, 
        because merging caption (including padding) is not supported in default.
        Args:
            data: list of tuple (tokens, stacked_token_vector_tensor, chunk_tags, pos_tags, ner_tags_tensor).
        Returns:
            tuple of reshaped and collated (tokens, stacked_token_vector_tensor, chunk_tags, pos_tags, ner_tags_tensor, seq_lengths).
        """
        data.sort(key=lambda item: len(item[0]), reverse=True)

        batch_size = len(data)
        seq_lengths = [len(item[0]) for item in data]
        max_seq_length = max(seq_lengths)

        for i, seq_len in enumerate(seq_lengths):
            padding_length = max_seq_length - seq_len

            padded_tokens = [padding_token] * (padding_length)
            padded_token_vectors = [padding_token_embedding] * (padding_length)
            padded_chunk_tags = [padding_token] * (padding_length)
            padded_pos_tags = [padding_token] * (padding_length)
            padded_ner_tags = [padding_ner_tag] * (padding_length)

            data[i][0].extend(padded_tokens) # TOKEN
            data[i][1].extend(padded_token_vectors) # TOKEN_EMBEDDING
            data[i][2].extend(padded_chunk_tags) # CHUNK_TAG
            data[i][3].extend(padded_pos_tags) # POS_TAG
            data[i][4].extend(padded_ner_tags) # NER_TAG

        # r"""Puts each data field into a tensor with outer dimension batch size"""
        transposed = list(zip(*data))

        collated_data = (
            transposed[0], # TOKEN
            torch.stack([torch.stack(item) for item in transposed[1]]), # TOKEN_EMBEDDING
            transposed[2], # CHUNK_TAG
            transposed[3], # POS_TAG
            torch.tensor(list(transposed[4]), requires_grad=False), # NER_TAG
            torch.tensor(seq_lengths, requires_grad=False)
        )

        collated_data[1].requires_grad_(False)
        collated_data[4].requires_grad_(False)
        collated_data[5].requires_grad_(False)

        return collated_data

    return collate_to_minibatch

def get_aida_yago_collater(word_embeddings, word_embeddings_dimension, pad_first=False, padding_token='<PAD>'):
    padding_token_embedding = np.random.uniform(-0.25, 0.25, word_embeddings_dimension).astype('f')
    if padding_token not in word_embeddings.vocab:
        word_embeddings[padding_token] = padding_token_embedding
    
    padding_token_embedding = torch.tensor(padding_token_embedding, dtype=torch.float32, requires_grad=False)
    padding_ner_tag = get_ner_tag_switcher()[padding_token]

    padding_token_embedding.requires_grad_(False)

    def collate_to_minibatch(data):
        """Creates mini-batch tensors from the list of tuples (image, caption).
        
        We should build custom collate_fn rather than using default collate_fn, 
        because merging caption (including padding) is not supported in default.
        Args:
            data: list of tuple (tokens, stacked_token_vector_tensor, wikipedia_links, complete_entities).
        Returns:
            tuple of reshaped and collated (tokens, stacked_token_vector_tensor, wikipedia_links, complete_entities, seq_lengths).
        """
        data.sort(key=lambda item: len(item[0]), reverse=True)

        batch_size = len(data)
        seq_lengths = [len(item[0]) for item in data]
        max_seq_length = max(seq_lengths)

        for i, seq_len in enumerate(seq_lengths):
            padding_length = max_seq_length - seq_len

            padded_tokens = [padding_token] * (padding_length)
            padded_token_vectors = [padding_token_embedding] * (padding_length)
            padded_wikipedia_links = [padding_token] * (padding_length)
            padded_complete_entities = [padding_token] * (padding_length)

            data[i][0].extend(padded_tokens) # TOKEN
            data[i][1].extend(padded_token_vectors) # TOKEN_EMBEDDING
            data[i][2].extend(padded_wikipedia_links) # WIKIPEDIA_LINKS
            data[i][3].extend(padded_complete_entities) # COMPLETE_ENTITIES

        # r"""Puts each data field into a tensor with outer dimension batch size"""
        transposed = list(zip(*data))

        collated_data = (
            transposed[0], # TOKEN
            torch.stack([torch.stack(item) for item in transposed[1]]), # TOKEN_EMBEDDING
            transposed[2], # WIKIPEDIA_LINKS
            transposed[3], # COMPLETE_ENTITIES
            torch.tensor(seq_lengths,requires_grad=False)
        )

        collated_data[1].requires_grad_(False)
        collated_data[4].requires_grad_(False)

        return collated_data

    return collate_to_minibatch

def load_ulzee_embeddings(data_root_dir):
    word_embedding_file = open(data_root_dir + '/word_embeds.npy', 'rb')
    word_list_file = open(data_root_dir + '/word_list.txt')
    
    zipped_kv = zip(list(map(lambda key: str.strip(key), word_list_file)), list(np.load(word_embedding_file)))

    word_embeddings = NumpyWordVector(zipped_kv)
    
    word_embedding_file.close()
    word_list_file.close()

    return word_embeddings

def report_results(filename, sentences, true_tags, predicted_tags):
  f = open(filename, "w")
  for sentenceIndex in range(len(true_tags)):
    for tagIndex in range(len(true_tags[sentenceIndex])):
      f.write('%s\t%s\t%s\n' % (sentences[sentenceIndex][tagIndex], true_tags[sentenceIndex][tagIndex], predicted_tags[sentenceIndex][tagIndex]))
    f.write('\n')
  f.close()

def report_results_aida_yago(filename, wikipedia_link_predictions):
  f = open(filename, "w")
  for entity in wikipedia_link_predictions.keys():
    for entity_predictions in wikipedia_link_predictions[entity]:
      if entity is not None and entity != '' and entity_predictions[0] is not None and entity_predictions[0] != '' and entity_predictions[1] is not None and entity_predictions[1] != '':
        f.write('%s\t%s\t%s\n' % (entity, entity_predictions[0], entity_predictions[1]))
  f.close()

def collate_predicted_data(tokens, predicted_y, actual_y, padding_token='<PAD>'):
    # tokens tuple -> (minibatch_size, sequence_length)

    # (minibatch_size, sequence_length)
    # flatten all the labels
    actual_y = actual_y.contiguous()
    actual_y = actual_y.view(-1)

    # (minibatch_size, sequence_length, num_classes) -> (minibatch_size, sequence_length)
    # Get one-hot predictions.
    _, predicted_y = torch.max(predicted_y, dim=2, keepdim=True)
    
    # flatten all predictions
    predicted_y = predicted_y.contiguous()
    predicted_y = predicted_y.view(-1)

    # create a mask by filtering out all tokens that ARE NOT the padding token
    tag_pad_token = get_ner_tag_switcher()[padding_token]
    mask = (actual_y != tag_pad_token).float()

    sentences = []
    predicted_tags = []
    true_tags = []

    reverse_ner_tag_switcher = get_reverse_ner_tag_switcher()

    numSentences = len(tokens)
    numTokensPerSentence = len(tokens[0])
    for item_idx in range(numSentences):
        tokens_in_sentence = tokens[item_idx]
        predicted_y_of_sentence = predicted_y[item_idx * numTokensPerSentence : (item_idx + 1) * numTokensPerSentence]
        actual_y_of_sentence = actual_y[item_idx * numTokensPerSentence : (item_idx + 1) * numTokensPerSentence]
        mask_of_sentence = mask[item_idx * numTokensPerSentence : (item_idx + 1) * numTokensPerSentence]

        sentence_token_compressed = []
        predicted_y_compressed = []
        actual_y_compressed = []
        for token_index in range(numTokensPerSentence):
            if mask_of_sentence[token_index] == 0:
                break
            
            sentence_token_compressed.append(tokens_in_sentence[token_index])
            predicted_y_compressed.append(reverse_ner_tag_switcher[predicted_y_of_sentence[token_index].item()])
            actual_y_compressed.append(reverse_ner_tag_switcher[actual_y_of_sentence[token_index].item()])

        sentences.append(sentence_token_compressed)
        predicted_tags.append(predicted_y_compressed)
        true_tags.append(actual_y_compressed)

    return sentences, predicted_tags, true_tags

def get_mask_from_text(wikipedia_links, padding_token):
    mask = []
    for sentence in wikipedia_links:
        sentence_mask = []
        for link in sentence:
            if link == '<PAD>':
                sentence_mask.append(0.0)
            else:
                sentence_mask.append(1.0)
        mask.append(sentence_mask)
    return torch.tensor(mask, dtype=torch.float32, requires_grad=False)

def addToDict(entities, previousEntity, previousClass, previousWikipediaLink):
    if previousEntity in entities:
        if previousClass in entities[previousEntity]:
            if previousWikipediaLink in entities[previousEntity][previousClass]:
                entities[previousEntity][previousClass][previousWikipediaLink] += 1
            else:
                entities[previousEntity][previousClass][previousWikipediaLink] = 1
        else:
            entities[previousEntity][previousClass] = {previousWikipediaLink: 1}
    else:
        entities[previousEntity] = {previousClass: {previousWikipediaLink: 1}}

def addToPredictionDict(output_entities, all_entity_links, previousEntity, previousClass, previousWikipediaLink):
    if previousEntity not in all_entity_links or previousClass not in all_entity_links[previousEntity]:
        return
    all_links = all_entity_links[previousEntity][previousClass]
    predicted_wikilink = max(all_links.items(), key=operator.itemgetter(1))[0]
    if previousEntity not in output_entities:
        output_entities[previousEntity] = [(predicted_wikilink, previousWikipediaLink)]
    else:
        output_entities[previousEntity].append((predicted_wikilink, previousWikipediaLink))

# Bayesian training
def collate_aida_yago_predicted_data(tokens, predicted_y, wikipedia_links, complete_entities, padding_token='<PAD>'):
    # tokens tuple -> (minibatch_size, sequence_length)

    # (minibatch_size, sequence_length)
    # flatten all the labels
    # actual_y = actual_y.contiguous()
    # actual_y = actual_y.view(-1)

    # (minibatch_size, sequence_length, num_classes) -> (minibatch_size, sequence_length)
    # Get one-hot predictions.
    _, predicted_y = torch.max(predicted_y, dim=2, keepdim=True)

    # flatten all predictions
    # predicted_y = predicted_y.contiguous()
    # predicted_y = predicted_y.view(-1)

    # create a mask by filtering out all tokens that ARE NOT the padding token
    tag_pad_token = get_ner_tag_switcher()[padding_token]
    mask = get_mask_from_text(wikipedia_links, padding_token)

    sentences = []
    predicted_tags = []
    true_tags = []

    reverse_ner_tag_switcher = get_reverse_ner_tag_switcher()

    numSentences = len(tokens)
    numTokensPerSentence = len(tokens[0])
    entities = {}
    for item_idx in range(numSentences):
        tokens_in_sentence = tokens[item_idx]
        predicted_y_of_sentence = predicted_y[item_idx]
        wikipedia_links_of_sentence = wikipedia_links[item_idx]
        mask_of_sentence = mask[item_idx]

        sentence_token_compressed = []
        predicted_y_compressed = []

        # previousEntity = ''
        # previousClass = tag_pad_token
        # previousWikipediaLink = ''
        for token_index in range(numTokensPerSentence):
            if mask_of_sentence[token_index] == 0:
                # If we see a padding token, everything after this token will be padding tokens 
                break
            
            token = tokens_in_sentence[token_index].strip()
            ner_tag = predicted_y_of_sentence[token_index].item()
            wikipediaLink = wikipedia_links_of_sentence[token_index]

            addToDict(entities, token, ner_tag, wikipediaLink)
            
        #     # Only add to list if previous class is different 
        #     if previousClass != predicted_y_of_sentence[token_index].item():
        #         if previousEntity != '' and previousWikipediaLink != '':
        #             # New entity found
        #             addToDict(entities, previousEntity, previousClass, previousWikipediaLink)
        #             previousEntity = ''
        #     else:
        #         # Continuing same class append to previous entity
        #         previousEntity += ' ' + tokens_in_sentence[token_index].strip()

        #     previousClass = predicted_y_of_sentence[token_index].item()

        #     if wikipedia_links_of_sentence[token_index] != '':
        #         previousWikipediaLink = wikipedia_links_of_sentence[token_index]

        # if previousEntity != '' and previousWikipediaLink != '':
        #     previousEntity = str.strip(previousEntity)
        #     addToDict(entities, previousEntity, previousClass, previousWikipediaLink)
        #     previousEntity = ''

    return entities

# Bayesian evaluation
def compute_aida_yago_predicted_data(tokens, predicted_y, wikipedia_links, complete_entities, all_entity_links, padding_token='<PAD>'):
    # tokens tuple -> (minibatch_size, sequence_length)

    # (minibatch_size, sequence_length)
    # flatten all the labels
    # actual_y = actual_y.contiguous()
    # actual_y = actual_y.view(-1)

    # (minibatch_size, sequence_length, num_classes) -> (minibatch_size, sequence_length)
    # Get one-hot predictions.
    _, predicted_y = torch.max(predicted_y, dim=2, keepdim=True)

    # flatten all predictions
    # predicted_y = predicted_y.contiguous()
    # predicted_y = predicted_y.view(-1)

    # create a mask by filtering out all tokens that ARE NOT the padding token
    tag_pad_token = get_ner_tag_switcher()[padding_token]
    mask = get_mask_from_text(wikipedia_links, padding_token)

    sentences = []
    predicted_tags = []
    true_tags = []

    reverse_ner_tag_switcher = get_reverse_ner_tag_switcher()

    numSentences = len(tokens)
    numTokensPerSentence = len(tokens[0])
    output_entities = {}
    
    for item_idx in range(numSentences):
        tokens_in_sentence = tokens[item_idx]
        predicted_y_of_sentence = predicted_y[item_idx]
        wikipedia_links_of_sentence = wikipedia_links[item_idx]
        mask_of_sentence = mask[item_idx]

        sentence_token_compressed = []
        predicted_y_compressed = []

        # previousEntity = ''
        # previousClass = tag_pad_token
        # previousWikipediaLink = ''
        for token_index in range(numTokensPerSentence):
            if mask_of_sentence[token_index] == 0:
                # If we see a padding token, everything after this token will be padding tokens 
                break

            token = tokens_in_sentence[token_index].strip()
            ner_tag = predicted_y_of_sentence[token_index].item()
            wikipediaLink = wikipedia_links_of_sentence[token_index]

            addToPredictionDict(output_entities, all_entity_links, token, ner_tag, wikipediaLink)

        #     # Only add to list if previous class is 
        #     if previousClass != predicted_y_of_sentence[token_index].item():
        #         if previousEntity != '' and previousWikipediaLink != '':
        #             # New entity found
        #             addToPredictionDict(output_entities, all_entity_links, previousEntity, previousClass, previousWikipediaLink)
        #             previousEntity = ''
        #     else:
        #         # Continuing same class append to previous entity
        #         previousEntity += ' ' + tokens_in_sentence[token_index].strip()

        #     previousClass = predicted_y_of_sentence[token_index].item()
            
        #     if wikipedia_links_of_sentence[token_index] != '':
        #         previousWikipediaLink = wikipedia_links_of_sentence[token_index]
        
        # if previousEntity != '' and previousWikipediaLink != '':
        #     # New entity found
        #     previousEntity = str.strip(previousEntity)
        #     addToPredictionDict(output_entities, all_entity_links, previousEntity, previousClass, previousWikipediaLink)
        #     previousEntity = ''

    return output_entities

def merge_aida_yago_entities(all_entity_links, batch_entities):
    for entity in batch_entities.keys():
        if entity not in all_entity_links:
            all_entity_links[entity] = batch_entities[entity]
        else:
            for tag in batch_entities[entity].keys():
                if tag not in all_entity_links[entity]:
                    all_entity_links[entity][tag] = batch_entities[entity][tag]
                else:
                    for wikipedia_link in batch_entities[entity][tag].keys():
                        if wikipedia_link not in all_entity_links[entity][tag]:
                            all_entity_links[entity][tag][wikipedia_link] = batch_entities[entity][tag][wikipedia_link]
                        else:
                            all_entity_links[entity][tag][wikipedia_link] += batch_entities[entity][tag][wikipedia_link]
    return all_entity_links

def merge_aida_yago_predictions(all_wikipedia_link_predictions, batch_wikipedia_link_predictions):
    for entity in batch_wikipedia_link_predictions.keys():
        if entity not in all_wikipedia_link_predictions:
            all_wikipedia_link_predictions[entity] = batch_wikipedia_link_predictions[entity]
        else:
            all_wikipedia_link_predictions[entity].extend(batch_wikipedia_link_predictions[entity])
    return all_wikipedia_link_predictions

def compute_and_report_results(model, dataloader, outfile='', padding_token='<PAD>', device=torch.device('cpu')):
    sentences = []
    predicted_tags = []
    true_tags = []
    for batch_index, sample in enumerate(dataloader):     # gives batch data, normalize x when iterate train_loader
        tokens = sample[0]
        tokens_vectors = sample[1].to(device)
        chunk_tags = sample[2]
        pos_tags = sample[3]
        ner_tags = sample[4].to(device)
        seq_lengths = sample[5].to(device)
        y = model(tokens_vectors, seq_lengths)              # model output

        sentences_batch, predicted_tags_batch, true_tags_batch = collate_predicted_data(tokens, y, ner_tags, padding_token)
        
        sentences.extend(sentences_batch)
        predicted_tags.extend(predicted_tags_batch)
        true_tags.extend(true_tags_batch)

    report_results(outfile, sentences, true_tags, predicted_tags)

def compute_and_report_aida_yago_results(model, dataloader, outfile='', padding_token='<PAD>', device=torch.device('cpu')):
    sentences = []
    predicted_tags = []

    all_entity_links = {}
    all_wikipedia_link_predictions = {}

    # Calculating Bayesian probability
    for batch_index, sample in enumerate(dataloader):     # gives batch data, normalize x when iterate train_loader
        tokens = sample[0]
        tokens_vectors = sample[1].to(device)
        wikipedia_links = sample[2]
        complete_entities = sample[3]
        seq_lengths = sample[4].to(device)
        y = model(tokens_vectors, seq_lengths)              # model output

        batch_entities = collate_aida_yago_predicted_data(tokens, y, wikipedia_links, complete_entities, padding_token)
        all_entity_links = merge_aida_yago_entities(all_entity_links, batch_entities)

    # Testing by Bayesian approach
    for batch_index, sample in enumerate(dataloader):     # gives batch data, normalize x when iterate train_loader
        tokens = sample[0]
        tokens_vectors = sample[1].to(device)
        wikipedia_links = sample[2]
        complete_entities = sample[3]
        seq_lengths = sample[4].to(device)
        y = model(tokens_vectors, seq_lengths)              # model output

        wikipedia_link_predictions = compute_aida_yago_predicted_data(tokens, y, wikipedia_links, complete_entities, all_entity_links, padding_token)
        all_wikipedia_link_predictions = merge_aida_yago_predictions(all_wikipedia_link_predictions, wikipedia_link_predictions)

    report_results_aida_yago(outfile, all_wikipedia_link_predictions)

def pytorch_cross_entropy(y, ner_tags, loss_func, device=torch.device('cpu')):
    # (1, 9, 9) -> (9, 9)
    # (batch_size, seq_length, vector_features) -> (batch_size * seq_length, vector_features)
    # y = y.contiguous()
    # y = y.view(y.shape[0]*y.shape[1], -1).to(device=device)
    
    # (1, 9) -> (9)
    # (batch_size, seq_len) -> (batch_size * seq_len)
    # ner_tags = ner_tags.contiguous()
    # ner_tags = ner_tags.view(ner_tags.shape[0]*ner_tags.shape[1]).to(device=device)

    y = y.permute(0, 2, 1)
    return loss_func(y, ner_tags)

if __name__ == "__main__":
    DEFAULT_BASE_PATH='/scratch/nmt324/BDML/Assignment 2/'
    DEFAULT_OPTIMIZER='adam'
    DEFAULT_USE_CUDA=False
    DEFAULT_NUM_WORKERS=1

    parser = argparse.ArgumentParser(description='Train and test a CNN to detect high-level features.')
    parser.add_argument('-c', '--USE_CUDA', help=f'To use cuda or not (default: {DEFAULT_USE_CUDA})', action='store_true', default=DEFAULT_USE_CUDA)
    parser.add_argument('-d', '--DATA_PATH', help=f'Location of data (default: {DEFAULT_BASE_PATH})', default=DEFAULT_BASE_PATH)
    parser.add_argument('-n', '--NUM_WORKERS', help=f'Number of workers (default: {DEFAULT_NUM_WORKERS})', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('-o', '--optimizer', help=f'Optimizer to use (default: {DEFAULT_OPTIMIZER})', type=Optimizer, choices=list(Optimizer), default=DEFAULT_OPTIMIZER)

    args = vars(parser.parse_args())

    TRAIN_BATCH_SIZE = 256
    VALIDATION_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 256
    AIDA_YAGO_BATCH_SIZE = 256

    NUM_EPOCH = 5

    MODEL_NUMBER = '95'

    NUM_WORKERS_TRAIN = args['NUM_WORKERS']
    NUM_WORKERS_VALIDATION = args['NUM_WORKERS']
    NUM_WORKERS_TEST = args['NUM_WORKERS']
    NUM_WORKERS_AIDA_YAGO = args['NUM_WORKERS']

    print(MODEL_NUMBER)
    print(args)

    PIN_MEMORY = False

    HOST_DEVICE = torch.device('cpu')
    if args['USE_CUDA'] and torch.cuda.is_available():
        HOST_DEVICE = torch.device('cuda')
        PIN_MEMORY = True

    TRAIN_DATA_PATH = args['DATA_PATH'] + '/preprocess/train_data'
    VALIDATION_DATA_PATH = args['DATA_PATH'] + '/preprocess/testa_data'
    TEST_DATA_PATH = args['DATA_PATH'] + '/preprocess/testb_data'
    AIDA_YAGO_DATA_PATH = args['DATA_PATH'] + '/preprocess/aida_yago_wikipedia'
    MODEL_FOLDER = args['DATA_PATH'] + '/models/'
    TRAIN_CONTROL_FILENAME = args['DATA_PATH'] + '/train_control_' + MODEL_NUMBER

    train_control_file = open(TRAIN_CONTROL_FILENAME, "w")
    train_control_file.write('1')
    train_control_file.close()

    load_google_embedding = False

    word_embeddings = None
    word_embeddings_dimension = 300

    start_embedding_load = time.monotonic()
    if load_google_embedding:
        # Load Google's pre-trained Word2Vec model.
        VOCAB_LIMIT = 5000 # None
        word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(args['DATA_PATH'] + '/GoogleNews-vectors-negative300.bin', binary=True, limit=VOCAB_LIMIT)
        word_embeddings_dimension = word_embeddings.vector_size
    else:
        # Load Ulzee's word embeddings
        word_embeddings = load_ulzee_embeddings(args['DATA_PATH'])
        word_embeddings_dimension = word_embeddings.vector_size

    end_embedding_load = time.monotonic()

    print('Time to load embedding: %s' % (end_embedding_load - start_embedding_load))

    padding_token = '<PAD>'
    collate_to_minibatch = get_collater(word_embeddings, word_embeddings_dimension, pad_first=True, padding_token=padding_token)
    collate_aida_yago_to_minibatch = get_aida_yago_collater(word_embeddings, word_embeddings_dimension, pad_first=True, padding_token=padding_token)

    conll_2003_train = CONLL_2003_Dataset(word_embeddings=word_embeddings, data_root_dir=TRAIN_DATA_PATH, device=HOST_DEVICE)
    conll_2003_validation = CONLL_2003_Dataset(word_embeddings=word_embeddings, data_root_dir=VALIDATION_DATA_PATH, device=HOST_DEVICE)
    conll_2003_test = CONLL_2003_Dataset(word_embeddings=word_embeddings, data_root_dir=TEST_DATA_PATH, device=HOST_DEVICE)
    aida_yago = AIDA_YAGO_Dataset(word_embeddings=word_embeddings, data_root_dir=AIDA_YAGO_DATA_PATH, device=HOST_DEVICE)

    print('Set up dataset')

    train_loader = DataLoader(dataset=conll_2003_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_TRAIN, collate_fn=collate_to_minibatch, pin_memory=PIN_MEMORY)
    validation_loader = DataLoader(dataset=conll_2003_validation, batch_size=VALIDATION_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_VALIDATION, collate_fn=collate_to_minibatch, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(dataset=conll_2003_test, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_TEST, collate_fn=collate_to_minibatch, pin_memory=PIN_MEMORY)
    aida_yago_loader = DataLoader(dataset=aida_yago, batch_size=AIDA_YAGO_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_AIDA_YAGO, collate_fn=collate_aida_yago_to_minibatch, pin_memory=PIN_MEMORY)

    print('Set up dataloader')

    ner_tag_switcher = get_ner_tag_switcher()
    output_classes = len(ner_tag_switcher) - 1                  # Subtracting 1 to account for the padding token's class
    model = NER_LSTM_BiDi(vector_dimension=word_embeddings_dimension, output_size=output_classes, batch_size=TRAIN_BATCH_SIZE, padding_token=padding_token).to(device=HOST_DEVICE)

    print('Set up model')

    optimizer = get_optimizer(args['optimizer'], model)                 # optimize all model parameters

    cross_entropy_loss_func = torch.nn.CrossEntropyLoss(ignore_index=ner_tag_switcher[padding_token]).to(HOST_DEVICE)             # the target label is one-hotted

    print('Set up loss')

    dataloading_time = 0
    dataloading_count = 0
    minibatch_time = 0
    minibatch_count = 0
    epoch_time = 0

    avg_loss = np.zeros(NUM_EPOCH, dtype=np.float32)
    avg_loss_validation = np.zeros(NUM_EPOCH, dtype=np.float32)
    epochs = 0
    for epoch in range(NUM_EPOCH):
        start_epoch = time.monotonic()
        start_dataloading = time.monotonic()

        epoch_loss = 0.0
        num_iterations_per_epoch = 0
        for batch_index, sample in enumerate(train_loader):     # gives batch data, normalize x when iterate train_loader
            end_dataloading = time.monotonic()
            tokens = sample[0]
            tokens_vectors = sample[1].to(HOST_DEVICE)
            chunk_tags = sample[2]
            pos_tags = sample[3]
            ner_tags = sample[4].to(HOST_DEVICE)
            seq_lengths = sample[5].to(HOST_DEVICE)
            y = model(tokens_vectors, seq_lengths)              # model output

            loss = pytorch_cross_entropy(y, ner_tags, cross_entropy_loss_func, device=HOST_DEVICE)
            optimizer.zero_grad()                               # clear gradients for this training step
            loss.backward()                                     # backpropagation, compute gradients
            optimizer.step()                                    # apply gradients

            end_minibatch = time.monotonic()

            print("Epoch: %s\nBatch: %s\nLoss: %s\n\n" % (epoch, batch_index, loss.item()))

            dataloading_time += end_dataloading - start_dataloading
            dataloading_count += 1
            minibatch_time += end_minibatch - start_dataloading
            minibatch_count += 1
            start_dataloading = time.monotonic()
            epoch_loss += loss
            num_iterations_per_epoch += 1

        epoch_loss_validation = 0.0
        num_iterations_per_epoch_validation = 0
        for batch_index, sample in enumerate(validation_loader):     # gives batch data, normalize x when iterate train_loader
            end_dataloading = time.monotonic()
            tokens = sample[0]
            tokens_vectors = sample[1].to(HOST_DEVICE)
            chunk_tags = sample[2]
            pos_tags = sample[3]
            ner_tags = sample[4].to(HOST_DEVICE)
            seq_lengths = sample[5].to(HOST_DEVICE)
            y = model(tokens_vectors, seq_lengths)              # model output

            loss = pytorch_cross_entropy(y, ner_tags, cross_entropy_loss_func, device=HOST_DEVICE)

            epoch_loss_validation += loss
            num_iterations_per_epoch_validation += 1
            print("Epoch: %s\nBatch: %s\nLoss: %s\n\n" % (epoch, batch_index, loss.item()))

        avg_loss[epoch] = epoch_loss/num_iterations_per_epoch
        avg_loss_validation[epoch] = epoch_loss_validation/num_iterations_per_epoch_validation

        print('Average epoch loss: %s' % avg_loss)
        print('Average validation epoch loss: %s' % avg_loss_validation)

        end_epoch = time.monotonic()
        epoch_time += end_epoch - start_epoch
        epochs += 1

        train_control_file = open(TRAIN_CONTROL_FILENAME, "r")
        fileContents = train_control_file.read()
        train_control_file.close()

        # External switch to stop training at the end of an epoch
        if '0' in fileContents:
            break

    print('Average epoch loss: %s' % avg_loss)
    print('Aggregated time spent waiting to load the batch from the DataLoader during the training: %s' % (dataloading_time / dataloading_count))
    print('Aggregated time for a mini-batch computation: %s' % (minibatch_time / minibatch_count))
    print('Aggregated time for each epoch: %s' % (epoch_time / NUM_EPOCH))
    
    model.eval()
    compute_and_report_results(model, train_loader, outfile ='./output_file_pytorch.train_'+MODEL_NUMBER, padding_token=padding_token, device=HOST_DEVICE)
    compute_and_report_results(model, validation_loader, outfile ='./output_file_pytorch.testa_'+MODEL_NUMBER, padding_token=padding_token, device=HOST_DEVICE)
    compute_and_report_results(model, test_loader, outfile ='./output_file_pytorch.testb_'+MODEL_NUMBER, padding_token=padding_token, device=HOST_DEVICE)
    
    if os.path.exists(MODEL_FOLDER):
        print('%s exists. Skipping it' % MODEL_FOLDER)
    else:
        os.mkdir(MODEL_FOLDER)
    
    # torch.save(model.state_dict(), MODEL_FOLDER + 'model_'+MODEL_NUMBER)
    # model.load_state_dict(torch.load(MODEL_FOLDER + 'model_'+MODEL_NUMBER))
    compute_and_report_aida_yago_results(model, aida_yago_loader, outfile ='./output_file_pytorch.aida_yago_'+MODEL_NUMBER, padding_token=padding_token, device=HOST_DEVICE)
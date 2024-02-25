#!/usr/bin/env python3

import torch 
from typing import List
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json
from collections import Counter

from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab, Vectors

from tqdm import tqdm
import numpy as np

import scipy.sparse as sp
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dataclasses import dataclass
from sklearn.metrics import confusion_matrix



# ------------------------------------------------
# The code provided in this file is taken from: https://github.com/andreabac3/NLP-Semantic-Role-Labeling
# and adapted to our purposes 
# ------------------------------------------------


#torch.cuda.empty_cache()


# CONSTANTS
USE_STORED: bool = False  # if true the train will not performed, we use the pre-trained model
USE_STORED_DATASET: bool = False  # if true will be used the pre calculated dataset
USE_GPU: bool = False #True  # if false, turn off the GPU
USE_TEST_EVALUATION: bool = False #True
SEED: int = 42  # fix the seed for reproducibility reason

USE_GLOVE: bool = True  # if true, the glove embeddings will loaded.
USE_BERT_EMBEDDINGS: bool = True  # if true, the bert embeddings will be calculated
USE_DEPENDENCY_HEADS: bool = True  # if true, the GCN will be used
USE_BIAFFINE_LAYER: bool = True  # if true, the attention layer will be used

USE_CRF: bool = False  # not used
USE_SYNTAGNET: bool = False  # not used

MIN_FREQUENCY: int = 2

EPOCHS: int = 20  # 11

BATCH_SIZE: int = 1 #16 #16 #64 #128  # 64 # 128
BATCH_SIZE_TEST: int = 1 #16 # 32
BATCH_SIZE_VALID_TEST: int = 1 #16 #16 #32

MAX_LEN: int = 143  # an integer to indicate the pre calculated max length.

PLOT_LOSS: bool = True  # if true, at the end of training you can see loss vs epochs and f1 vs epochs plot

unk_token: str = '<unk>'
pad_token: str = '<pad>'

model_name: str = 'bert-base-multilingual-uncased' #'bert-base-cased'



def net_configurator(use_pretrained: bool, use_bert_embeddings: bool, use_dependecy_heads: bool, use_biaffine_layer: bool, use_crf: bool, use_predicates: bool, use_syntagnet: bool) -> dict:
    '''
    Return a config used in the dataset class and in the model class, in order to enable/disable some layer, improve code modularity
    '''
    config: dict = {"use_syntagnet": use_syntagnet, "use_predicates": use_predicates, "use_pretrained": use_pretrained, "use_bert_embeddings": use_bert_embeddings, "use_dependecy_heads": use_dependecy_heads, "use_biaffine_layer": use_biaffine_layer, "use_crf": use_crf}
    return config



def read_dataset(path: str):
    '''
    SOURCE CODE: Sapienza NLP group
    '''
    with open(path) as f:
        dataset = json.load(f)

    sentences, labels = {}, {}
    for sentence_id, sentence in dataset.items():
        sentence_id = int(sentence_id)
        sentences[sentence_id] = {
            'words': sentence['words'],
            'lemmas': sentence['lemmas'],
            'pos_tags': sentence['pos_tags'],
            'dependency_heads': [int(head) for head in sentence['dependency_heads']],
            'predicates': sentence['predicates'],
        }

        labels[sentence_id] = {
            'predicates': sentence['predicates'],
            'roles': {int(p): r for p, r in sentence['roles'].items()}
        }

    return sentences, labels


def build_vocab(dataset, feature_type: str, min_freq: int = 1, pad_token: str = '<pad>', unk_token: str = '<unk>') -> vocab:
    '''
    Build a vocabulary for a given vector
    '''
    counter: Counter = Counter()
    for i in tqdm(range(len(dataset.keys()))):
        list_of_feature = dataset[i][feature_type]
        for word in list_of_feature:
            if word != pad_token:
                counter[word] += 1
    return vocab(counter, specials=[pad_token, unk_token], min_freq=min_freq)


def build_vocab_roles(dataset, pad_token: str = '<pad>') -> vocab:
    '''
    Build a vocabulary for the roles label vector
    '''
    counter: Counter = Counter()
    for sentence_id in tqdm(range(len(dataset.keys()))):
        roles_dict = dataset[sentence_id]['roles']
        predicate_indices = roles_dict.keys()
        for pred_index in predicate_indices:
            for role in roles_dict[pred_index]:
                if role != pad_token:
                    counter[role] += 1
    return vocab(counter, specials=[pad_token])


def initialize_matrix(dim, no_arc=0):
    # to avoid cloning list
    return [no_arc] * dim



def adjacency_matrix(tree_list: List[str]):
    '''
    starting from a dependecy heads in which each list element points to his father
    return a adjacency_matrix.
    Added extra node 0 for the root and the self-loop
    The matrix is normalized as suggested in this article https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780
    The normalization consist in the multiplication of the adjacency matrix by it's inverse degree matrix

    Portion of this function is taken from the kipf repository https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
    '''
    tree_list = [0] + [int(elem) for elem in tree_list]  # I add the extra node 0 for the root
    arcs: dict = {i: elem for i, elem in enumerate(tree_list)}  # arcs {1: 0, 2: 2, 3: 3, 0: 0}

    matrix = [initialize_matrix(dim=len(tree_list)) for _ in range(len(tree_list))]
    for i in range(len(matrix)):
        j = arcs[i]
        matrix[i][j] = 1

    for i in range(len(matrix)):
        matrix[0][i] = 0
    A = np.matrix(matrix)
    # start kipf repo utils.py
    A = sp.csr_matrix(A)
    adj = A
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # convert the direct graph to it's indirect version
    adj = normalize(adj + sp.eye(adj.shape[0]))  # add the self loops and then normalize the adjancency matrix
    # end kipf repo utils.py
    return adj.todense()


def normalize(mx):
    '''
    For GCN the normalization of the adj matrix is required to avoid the known problems of vanishing or exploding gradients.
    I normalized the adjacency matrix with indirect edges and with self-loops by its inverse degree matrix.
    SOURCE CODE: https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
    '''
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def batch_adj_matrix(adj_matrix_list: torch.Tensor) -> torch.Tensor:
    '''
    :param adj_matrix_list: shape (batch_size, side_square, side_square)
    :return: a torch.Tensor with matrix belong his diagonal
    in order to support batches for the GCN it is necessary
    to create a matrix that will contain all the adjacency matrices in a batch along its diagonal.
    as shown: https://user-images.githubusercontent.com/7347296/34198790-eb5bec96-e56b-11e7-90d5-157800e042de.png
    In order to contribute to the open source world, I created a pull request to add this function to the kipf (GCN creator) repository.
    Then I closed the pull request thinking it was better to wait for the end of the exam.
    pull request: https://github.com/tkipf/pygcn/pull/65
    '''
    dimension = adj_matrix_list.shape
    batch_size = dimension[0]
    side_of_the_square = dimension[2]
    side_batch_matrix = side_of_the_square * batch_size
    res_batch_matrix = torch.zeros((side_batch_matrix, side_batch_matrix))
    for batch_num in range(batch_size):
        res_batch_matrix[side_of_the_square * batch_num:side_of_the_square + (batch_num * side_of_the_square), side_of_the_square * batch_num:side_of_the_square + (batch_num * side_of_the_square)] = adj_matrix_list[batch_num]

    return res_batch_matrix



def evaluate_argument_classification(labels, predictions, null_tag='_'):
    '''
    SOURCE CODE: SAPIENZA NLP GROUP
    '''
    true_positives, false_positives, false_negatives = 0, 0, 0
    for sentence_id in labels:
        gold = labels[sentence_id]['roles']
        pred = predictions[sentence_id]['roles']
        predicate_indices = set(gold.keys()).union(pred.keys())

        for idx in predicate_indices:
            if idx in gold and idx not in pred:
                false_negatives += sum(1 for role in gold[idx] if role != null_tag)
            elif idx in pred and idx not in gold:
                false_positives += sum(1 for role in pred[idx] if role != null_tag)
            else:  # idx in both gold and pred
                for r_g, r_p in zip(gold[idx], pred[idx]):
                    if r_g != null_tag and r_p != null_tag:
                        if r_g == r_p:
                            true_positives += 1
                        else:
                            false_positives += 1
                            false_negatives += 1
                    elif r_g != null_tag and r_p == null_tag:
                        false_negatives += 1
                    elif r_g == null_tag and r_p != null_tag:
                        false_positives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_conf_matrix(cm, name_label, dataset_name: str):
    # standard function to plot confusion matrix
    cm_df = pd.DataFrame(cm, index=name_label, columns=name_label)
    plt.figure(figsize=(30, 30))
    sns.heatmap(cm_df, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Normalized Confusion Matrix ' + dataset_name)
    #plt.savefig('/confusion_matrix.png')
    plt.show()


def calculate_confusion_matrix(labels, predictions, dataset_name: str = '', include_placeholder: bool = True):
    # function used to plot confusion matrix
    pred_set = []
    label_set = []
    for sentence_id in labels:
        gold = labels[sentence_id]['roles']
        pred = predictions[sentence_id]['roles']
        pred_index_gold = gold.keys()
        for pred_id in pred_index_gold:
            for i in range(len(pred[pred_id])):
                if (pred[pred_id][i] == '_' or gold[pred_id][i] == '_') and include_placeholder:
                    # use to avoid the remove the placeholder tag from the confusion matrix
                    continue
                pred_set.append(pred[pred_id][i])
                label_set.append(gold[pred_id][i])
    name_label = list(set(label_set).union(set(pred_set)))
    cm = confusion_matrix(label_set, pred_set, labels=name_label, normalize='true')
    plot_conf_matrix(cm, name_label, dataset_name)
    return cm, name_label


def evaluate_argument_identification(labels, predictions, null_tag='_'):
    '''
    SOURCE CODE: SAPIENZA NLP GROUP
    '''
    true_positives, false_positives, false_negatives = 0, 0, 0
    for sentence_id in labels:
        gold = labels[sentence_id]['roles']
        pred = predictions[sentence_id]['roles']
        #print("gold", gold)
        #print("pred", pred)
        predicate_indices = set(gold.keys()).union(pred.keys())
        for idx in predicate_indices:
            if idx in gold and idx not in pred:
                false_negatives += sum(1 for role in gold[idx] if role != null_tag)
            elif idx in pred and idx not in gold:
                false_positives += sum(1 for role in pred[idx] if role != null_tag)
            else:  # idx in both gold and pred
                for r_g, r_p in zip(gold[idx], pred[idx]):
                    if r_g != null_tag and r_p != null_tag:
                        true_positives += 1
                    elif r_g != null_tag and r_p == null_tag:
                        false_negatives += 1
                    elif r_g == null_tag and r_p != null_tag:
                        false_positives += 1
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def print_output(model: nn.Module, dataloader_test: DataLoader, vocab_labels: vocab) -> dict:
    '''
    This function take a model and return prediction as request in the docker.
    This function support batch evaluation
    '''
    output_dict = dict()
    model.eval()
    with torch.no_grad():
        for sample in tqdm(dataloader_test):
            batch_size: int = len(sample['predicate_id'])
            predictions = model(sample)  # get batched predictions

            if model.use_crf:
                predictions = torch.LongTensor(model.crf.decode(predictions))
            else:
                predictions = torch.argmax(predictions, -1)

            for batch_num in range(batch_size):
                # build the output in the right format
                batch_predictions = predictions[batch_num].view(-1)

                mask_padding = sample['words'][batch_num] != 0  # removing padding
                encoded_predictions = batch_predictions[mask_padding]
                list_predictions = encoded_predictions.tolist()

                decode_predictions: List[str] = [vocab_labels.get_itos()[elem] for elem in list_predictions]  # back from vocabulary id List[int] to labels List[str]
                id_sentence: int = int(sample['sentence_id'][batch_num])
                index_pred: int = int(sample['predicate_id'][batch_num])
                if id_sentence in output_dict:
                    # add more than one list of roles at given id sentence
                    output_dict[id_sentence]['roles'][index_pred] = decode_predictions
                else:
                    # insert for the first time
                    output_dict[id_sentence] = {'roles': {index_pred: decode_predictions}}

        for id_sentence in dataloader_test.dataset.empty_predicates:
            # Case in which we don't have any index_pred
            output_dict[id_sentence] = {'roles': dict()}
        return output_dict


def plot_loss(data: dict) -> None:
    epochs_list = range(1, len(data['loss_train'])+1)
    loss_summary_train = data['loss_train']
    loss_summary_valid = data['loss_valid']
    f1_validation_summary_valid = data['f1_valid_identification']
    f1_classification_summary_valid = data['f1_valid_classification']

    plt.plot(epochs_list, loss_summary_train, label="Train")
    plt.plot(epochs_list, loss_summary_valid, label="Dev")
    best_loss_index = loss_summary_valid.index(min(loss_summary_valid))
    best_loss_value = loss_summary_valid[best_loss_index]
    f1_at_best_loss = f1_classification_summary_valid[best_loss_index]

    plt.axvline(x=best_loss_index + 1, label='Dev loss early stopping \n F1: {} \n Loss: {}'.format(round(f1_at_best_loss, 4), round(best_loss_value, 4)), c='red', linestyle=':')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Argument Classification Train vs Dev')
    plt.legend()
    plt.show()

    plt.plot(epochs_list, f1_validation_summary_valid, label="dev identification")
    plt.plot(epochs_list, f1_classification_summary_valid, label="dev classification")
    best_f1_index = f1_classification_summary_valid.index(max(f1_classification_summary_valid))
    best_f1_value = f1_classification_summary_valid[best_f1_index]
    loss_f1_best = loss_summary_valid[best_f1_index]
    plt.axvline(x=best_f1_index + 1, label='Classification f1 early stopping \n F1: {} \n Loss: {}'.format(round(best_f1_value, 4), round(loss_f1_best, 4)), c='red', linestyle=':')

    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.title('F1 Score Identification vs Classification')
    plt.legend()
    plt.show()





'''

BERT EMBEDDER
SOURCE CODE: SAPIENZA NLP GROUP

'''
class BERTEmbedder:

    def __init__(self, bert_model: AutoModel,
                 bert_tokenizer: AutoTokenizer,
                 device: str):
        """
        Args:
          bert_model (BertModel): The pretrained BERT model.
          bert_tokenizer (BertTokenizer): The pretrained BERT tokenizer.
          token_limit (integer): The maximum number of tokens to give as input to BERT.
          device (string): The device on which BERT should run, either cuda or cpu.
        """
        super(BERTEmbedder, self).__init__()
        self.bert_model = bert_model
        self.bert_model.to(device)
        self.bert_model.eval()
        self.bert_tokenizer = bert_tokenizer
        self.device = device

    def embed_sentences(self, sentences: List[str]):
        # we convert the sentences to an input that can be fed to BERT
        input_ids, to_merge_wordpieces, attention_mask, token_type_ids = self._prepare_input(sentences)
        # we set output_all_encoded_layers to True cause we want to sum the
        # representations of the last four hidden layers
        with torch.no_grad():
            # The BertModel forward method returns a tuple of 3 elements:
            # 1) last_hidden_states of shape (batch_size x sequence_length x hidden_size),
            # which is the sequence of hidden states of the last layer of the model,
            # 2) pooler_output of shape batch_size x hidden_size,
            # which is the hidden states of the first token of the sequence (the CLS token)
            # passed through a Linear layer with a Tanh activation function,
            # 3) hidden_states, which is a tuple of FloatTensors, each of shape
            # (batch_size x sequence_length x hidden_size), each FloatTensor is the hidden states
            # of the model at the output of one of BERT's layers.
            bert_output = self.bert_model.forward(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                  attention_mask=attention_mask)

        # we sum the sum of the last four hidden layers (-1 is the hidden states, see point (3) above)
        layers_to_sum = torch.stack([bert_output[-1][x] for x in [-1, -2, -3, -4]], axis=0)
        summed_layers = torch.sum(layers_to_sum, axis=0)
        merged_output = self._merge_embeddings(summed_layers, to_merge_wordpieces)

        return merged_output

    def _prepare_input(self, sentences: List[str]):
        input_ids = []
        # we must keep track of which words have been split so we can merge them afterwards
        to_merge_wordpieces = []
        # BERT requires the attention mask in order to know on which tokens it has to attend to
        # padded indices do not have to be attended to so will be 0
        attention_masks = []
        # BERT requires token type ids for doing sequence classification
        # in our case we do not need them so we set them all to 0
        token_type_ids = []
        # we sum 2 cause we have to consider also [CLS] and [SEP] in the sentence length
        max_len = max([len(self._tokenize_sentence(s)[0]) for s in sentences])
        for sentence in sentences:
            encoded_sentence, to_merge_wordpiece = self._tokenize_sentence(sentence)
            att_mask = [1] * len(encoded_sentence)
            att_mask = att_mask + [0] * (max_len - len(encoded_sentence))
            # we pad sentences shorter than the max length of the batch
            encoded_sentence = encoded_sentence + [0] * (max_len - len(encoded_sentence))
            input_ids.append(encoded_sentence)
            to_merge_wordpieces.append(to_merge_wordpiece)
            attention_masks.append(att_mask)
            token_type_ids.append([0] * len(encoded_sentence))
        input_ids = torch.LongTensor(input_ids).to(self.device)
        attention_masks = torch.LongTensor(attention_masks).to(self.device)
        token_type_ids = torch.LongTensor(token_type_ids).to(self.device)
        return input_ids, to_merge_wordpieces, attention_masks, token_type_ids

    def _tokenize_sentence(self, sentence: List[str]):
        encoded_sentence = [self.bert_tokenizer.cls_token_id]
        # each sentence must start with the special [CLS] token
        to_merge_wordpiece = []
        # we tokenize a word at the time so we can know which words are split into multiple subtokens
        for word in sentence:
            encoded_word = self.bert_tokenizer.tokenize(word)
            # we take note of the indices associated with the same word
            to_merge_wordpiece.append([i for i in range(len(encoded_sentence) - 1, len(encoded_sentence) + len(encoded_word) - 1)])
            encoded_sentence.extend(self.bert_tokenizer.convert_tokens_to_ids(encoded_word))
        # each sentence must end with the special [SEP] token
        encoded_sentence.append(self.bert_tokenizer.sep_token_id)
        return encoded_sentence, to_merge_wordpiece

    # aggregated_layers has shape: shape batch_size x sequence_length x hidden_size
    def _merge_embeddings(self, aggregated_layers: List[List[float]],
                          to_merge_wordpieces: List[List[int]]):
        merged_output = []
        # first we remove the [CLS] and [SEP] tokens from the output embeddings
        aggregated_layers = aggregated_layers[:, 1:-1, :]
        for embeddings, sentence_to_merge_wordpieces in zip(aggregated_layers, to_merge_wordpieces):
            sentence_output = []
            # for each word we retrieve the indices of its subtokens in the tokenized sentence
            for word_to_merge_wordpiece in sentence_to_merge_wordpieces:
                # we average all the embeddings of the subpieces of a word
                sentence_output.append(torch.mean(embeddings[word_to_merge_wordpiece], axis=0))
            merged_output.append(torch.stack(sentence_output).to(self.device))
        return merged_output




'''

SRL DATASET

'''
class SRL_Dataset(Dataset):

    def __init__(self, sentences: dict, labels: dict = None, device: str = None, pad_token: str = '<pad>', configurator: dict = None, max_len: int = None, bert_model=None, bert_tokenizer=None):
        super(SRL_Dataset, self).__init__()
        assert configurator is not None and device is not None
        self.device: str = device
        self.sentences: dict = sentences
        self.labels: dict = labels
        self.pad_token: str = pad_token
        self.empty_predicates: list = []
        self.max_len: int = max_len if max_len is not None else self._calculate_max_len(sentences=sentences)
        self.configurator: dict = configurator
        if self.configurator['use_bert_embeddings']:
            # bert embedder instantiation
            self.bert_emb: BERTEmbedder = BERTEmbedder(bert_model=bert_model, bert_tokenizer=bert_tokenizer, device=self.device)

    def _calculate_max_len(self, sentences) -> int:
        # calculate the max sentence length in the dataset
        max_len: int = max([len(sentences[id_sentence]['words']) for id_sentence in sentences.keys()])
        return max_len

    def encode_test(self, elem: List[str], vocab: vocab) -> List[int]:
        # make an encoding of the sentence.
        # from List of string to list of int using the vocabulary
        sample = []
        for i in range(len(elem)):
            if elem[i] not in vocab.get_stoi():
                sample.append(vocab['<unk>'])
                continue
            sample.append(vocab[elem[i]])
        return sample

    def encode_label(self, labels: List[str], vocab: vocab) -> List[int]:
        # encode label from class for instance Agent to -> label id
        return [vocab[label] for label in labels]

    def __len__(self) -> int:
        if self.samples is None:
            raise Exception("You should call build_sample()")
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self.samples is None:
            raise Exception("You should call build_sample()")
        return self.samples[idx]

    def right_pad_sentence(self, sentence: List[int], pad_token: int = 0) -> List[int]:
        '''
        Takes an encoded sentences and return the same sentence with a fixed pad length
        '''
        padded_sequence: List[int] = [pad_token] * self.max_len
        for i, word in enumerate(sentence):
            padded_sequence[i] = word
        return padded_sequence

    def _to_one_hot(self, enc_predicates: List[int], index: int, vocab_predicates, place_holder: str = '_') -> List[int]:
        '''
        Remove other predicates from a predicates list, I take only one predicates given his index.
        example
        [_,_,_,Agent,_,_,_,Topic], index = 3 --return--> [_,_,_,Agent,_,_,_,_]
        '''
        one_hot_predicates: List[int] = [vocab_predicates[place_holder]] * len(enc_predicates)
        one_hot_predicates[index] = enc_predicates[index]  # insert the right predicate at the given index
        return one_hot_predicates

    def build_sample(self, vocab_words, vocab_pos_tags, vocab_lemmas, vocab_predicates, vocab_label_roles, place_holder: str = '_'): # vocab_dependency_relations, place_holder: str = '_'):

        samples = []

        for sentence_id in tqdm(self.sentences.keys()):
            words_list = self.sentences[sentence_id]['words']
            len_words_list: int = len(words_list)
            pos_list: List[str] = self.sentences[sentence_id]['pos_tags']
            lemmas_list: List[str] = self.sentences[sentence_id]['lemmas']
            predicates_list: List[str] = self.sentences[sentence_id]['predicates']

            if set(predicates_list) == set('_'):
                # Skip all sentences without predicates and store their id
                self.empty_predicates.append(sentence_id)
                continue

            # -- Non trainable Bert Embeddings --
            if self.configurator['use_bert_embeddings']:
                # Calculate bert embeddings
                # I move the bert embeddings tensor to the CPU, I will move the bert embeddings to the right device during the train to avoid the gpu out of memory
                vector_bert_emb = self.bert_emb.embed_sentences([words_list])[0].to('cpu')
                diff: int = self.max_len - len_words_list
                pad_value: int = 0
                vector_bert_emb: torch.Tensor = torch.nn.functional.pad(vector_bert_emb.T, (0, diff), value=pad_value).T  # pad the bert embeddings
                vector_bert_emb: torch.FloatTensor = vector_bert_emb.to(torch.float16)  # In order to reduce the memory usage I move to the float16 representation

            # encoding of the sentence
            enc_words: List[int] = self.right_pad_sentence(self.encode_test(words_list, vocab_words), pad_token=vocab_words['<pad>'])
            enc_words: torch.LongTensor = torch.LongTensor(enc_words).to(self.device)

            enc_pos: List[int] = self.right_pad_sentence(self.encode_test(pos_list, vocab_pos_tags), pad_token=vocab_pos_tags['<pad>'])
            enc_pos: torch.LongTensor = torch.LongTensor(enc_pos).to(self.device)

            enc_lemmas: List[int] = self.right_pad_sentence(self.encode_test(lemmas_list, vocab_lemmas), pad_token=vocab_lemmas['<pad>'])
            enc_lemmas: torch.LongTensor = torch.LongTensor(enc_lemmas).to(self.device)

            
            if self.configurator['use_dependecy_heads']:
                # Create adjacency_matrix starting from dependency head (tree), then pad the adj matrix
                dependency_heads_list: List[str] = self.sentences[sentence_id]['dependency_heads']
                
                # Create the adjacency matrix, the matrix is already normalized
                dependency_heads_list: torch.tensor = torch.tensor(dependency_heads_list, device=self.device)

                # return a adjacency matrix
                dependency_heads_matrix: torch.FloatTensor = torch.FloatTensor(adjacency_matrix(dependency_heads_list)).to(self.device)  
                
                diff: int = self.max_len - len(dependency_heads_list) - 1
                pad_value: int = 0
                
                # Padding of the adjacency matrix
                heads_matrix_pad: torch.Tensor = torch.nn.functional.pad(dependency_heads_matrix, (0, diff), value=pad_value)

                heads_matrix_pad: torch.Tensor = torch.nn.functional.pad(heads_matrix_pad.T, (0, diff), value=pad_value).T
                dependency_heads_matrix = heads_matrix_pad

            enc_predicates: List[int] = self.encode_test(predicates_list, vocab_predicates)
            
            # return list of index in which the predicates are not place_holders
            predicate_indices: List[int] = [index for index, predicate in enumerate(predicates_list) if predicate != place_holder]  

            for id_predicates in predicate_indices:
                # there is at least one predicates
                one_hot_predicates: List[int] = self._to_one_hot(enc_predicates, id_predicates, vocab_predicates)  # [_,_,_,Agent,_,_,_,Topic], index = 3 --return--> [_,_,_,Agent,_,_,_,_]
                one_hot_predicates: List[int] = self.right_pad_sentence(one_hot_predicates, pad_token=vocab_predicates['<pad>'])
                one_hot_predicates: torch.LongTensor = torch.LongTensor(one_hot_predicates).to(self.device)

                sample: dict = {'words': enc_words, 'pos_tags': enc_pos, 'predicates': one_hot_predicates, 'lemmas': enc_lemmas,
                                'predicate_id': id_predicates, 'sentence_id': sentence_id, 
                                }

                if self.configurator['use_bert_embeddings']:
                    sample['bert_embeddings'] = vector_bert_emb
                if self.labels is not None:
                    print
                    label_list: List[int] = self.encode_label(self.labels[sentence_id]['roles'][id_predicates], vocab_label_roles)
                    padded_label: List[int] = self.right_pad_sentence(sentence=label_list, pad_token=vocab_label_roles['<pad>'])
                    label_list: torch.LongTensor = torch.LongTensor(padded_label).to(self.device)
                    sample['label'] = label_list
                if self.configurator['use_dependecy_heads']:
                    sample['dependency_heads_matrix'] = dependency_heads_matrix

                samples.append(sample)

        self.samples = samples
        return samples
    



'''

BIAFFINE ATTENTION
SOURCE CODE: GITHUB -> https://gist.github.com/JohnGiorgi/7472f3a523f53aed332ff2f8d6eff914

'''
class BiaffineAttention(torch.nn.Module):
    """Implements a biaffine attention operator for binary relation classification.
    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.
    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.
    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.
    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = torch.nn.Bilinear(in_features, in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(2 * in_features, out_features, bias=True)

        self.reset_parameters()

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()



'''

GRAPH CONVOLUTION
SOURCE CODE: https://github.com/meliketoy/graph-cnn.pytorch/blob/master/layers.py

'''
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    SOURCE CODE https://github.com/meliketoy/graph-cnn.pytorch/blob/master/layers.py
    """

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


'''

GRAPH CONVOLUTIONAL NETWORK

'''
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
    



'''

SEMANTIC ROLE LABELING MODEL

'''
class SRL_final_MODEL(nn.Module):
    def __init__(self, hparams, configurator: dict):
        super(SRL_final_MODEL, self).__init__()

        self.device: str = hparams.device

        self.dropout = nn.Dropout(hparams.dropout)

        self.real_lstm_hidden_dim: int = hparams.lstm_hidden_dim * 2 if hparams.bidirectional else hparams.lstm_hidden_dim

        self.lstm_input_size: int = hparams.embedding_dim_words + hparams.embedding_dim_pos + hparams.embedding_dim_lemmas + hparams.embedding_dim_predicates # hparams.embedding_dim_relations + hparams.embedding_dim_predicates

        self.word_embedding = nn.Embedding(hparams.vocab_size_words, hparams.embedding_dim_words, padding_idx=0)
        self.word_embedding_pos = nn.Embedding(hparams.vocab_size_pos_tags, hparams.embedding_dim_pos, padding_idx=0)
        self.lemma_embedding = nn.Embedding(hparams.vocab_size_lemmas, hparams.embedding_dim_lemmas, padding_idx=0)

        #self.dependency_relations_embedding = nn.Embedding(hparams.vocab_size_dependency_relations, hparams.embedding_dim_relations, padding_idx=0)

        self.predicates_embedding = nn.Embedding(hparams.vocab_size_predicates, hparams.embedding_dim_predicates, padding_idx=0)

        self.use_pretrained: bool = configurator['use_pretrained']
        if self.use_pretrained:
            # LOAD GloVe embeddings
            self.word_embedding.weight.data.copy_(hparams.glove_embeddings)

        # --- CRF Layer ---
        self.use_crf: bool = configurator['use_crf']
        if self.use_crf:
            print("we are using crf")
            self.crf = CRF(num_tags=hparams.num_classes, batch_first=True)

        # --- BERT EMB ---
        self.use_bert_embeddings: bool = configurator['use_bert_embeddings']
        if self.use_bert_embeddings:
            self.lstm_bert_emb = nn.LSTM(hparams.bert_hidden_dim, hparams.lstm_hidden_dim,
                                         bidirectional=hparams.bidirectional,
                                         batch_first=True,
                                         num_layers=hparams.bert_lstm_num_layers,
                                         dropout=hparams.lstm_dropout)
            self.lstm_input_size += self.real_lstm_hidden_dim

        # --- GCN Layer ---
        self.use_dependecy_heads: bool = configurator['use_dependecy_heads']
        if self.use_dependecy_heads:
            self.gcn_layer = GCN(nfeat=self.real_lstm_hidden_dim, nhid=hparams.gcn_hidden_dim, nclass=hparams.gcn_output_dim, dropout=hparams.gcn_dropout_probability)

            self.gcn_bilstm = nn.LSTM(self.lstm_input_size, hparams.lstm_hidden_dim,
                                      bidirectional=hparams.bidirectional,
                                      batch_first=True,
                                      num_layers=hparams.gcn_lstm_num_layers,
                                      dropout=hparams.lstm_dropout)
            self.lstm_input_size += hparams.gcn_output_dim

        self.lstm_emb = nn.LSTM(self.lstm_input_size, hparams.lstm_hidden_dim,
                                bidirectional=hparams.bidirectional,
                                batch_first=True,
                                num_layers=hparams.num_layers,
                                dropout=hparams.lstm_dropout)

        self.output_layer = nn.Linear(self.real_lstm_hidden_dim, hparams.num_classes)

        # --- Biaffine Layer ---
        self.use_biaffine_layer = configurator['use_biaffine_layer']
        if self.use_biaffine_layer:
            self.output_layer_2 = nn.Linear(self.real_lstm_hidden_dim, hparams.num_classes)
            self.bilstm_stacked = nn.LSTM(self.real_lstm_hidden_dim, hparams.lstm_hidden_dim,
                                          bidirectional=hparams.bidirectional,
                                          batch_first=True,
                                          num_layers=hparams.biaffine_lstm_num_layers,
                                          dropout=hparams.lstm_dropout)

            self.biaffine_scorer: BiaffineAttention = BiaffineAttention(hparams.num_classes, hparams.num_classes)
        self.use_predicate_biaffine = configurator['use_predicates'] and self.use_biaffine_layer
        if self.use_predicate_biaffine:
            self.biaffine_lstm_predicate = nn.LSTM(hparams.embedding_dim_predicates, hparams.lstm_hidden_dim,
                                                   bidirectional=hparams.bidirectional,
                                                   batch_first=True,
                                                   num_layers=hparams.biaffine_lstm_num_layers,
                                                   dropout=hparams.lstm_dropout)

    def forward(self, sample):
        x_word = sample['words']
        x_pos = sample['pos_tags']
        x_predicate = sample['predicates']
        x_lemma = sample['lemmas']

        dimension: torch.Size = x_word.shape
        batch_size: int = dimension[0]
        sequence_length: int = dimension[1]

        #  Produce the embeddings vectors starting from the encoded features
        word_emb: torch.Tensor = self.word_embedding(x_word)
        pos_emb: torch.Tensor = self.word_embedding_pos(x_pos)
        lemma_emb: torch.Tensor = self.lemma_embedding(x_lemma)
        predicates_emb = self.predicates_embedding(x_predicate)

        #relations_emb: torch.Tensor = self.dependency_relations_embedding(x_dependency_relation)
        word_representation: torch.Tensor = torch.cat((word_emb, pos_emb, lemma_emb, predicates_emb), dim=2) # relations_emb, predicates_emb), dim=2)  # core word representation

        if self.use_bert_embeddings:
            bert_embeddings = sample['bert_embeddings'].to(self.device)
            lstm_bert, _ = self.lstm_bert_emb(bert_embeddings.to(torch.float))
            bert_embeddings = bert_embeddings.to('cpu').to(torch.float16)  # Used only in gpu mode: move bert emb back to cpu.
            # I store the bert embeddings into float-16bit representation in order to save memory
            word_representation = torch.cat((word_representation, lstm_bert), dim=2)  # I add the bert emb to the core word repr.

        word_representation = self.dropout(word_representation)  # I apply the dropout to the word representation tensor (concatenation of different embeddings)

        if self.use_dependecy_heads:
            x_dependency_heads_matrix = sample['dependency_heads_matrix']
            gcn_bilstm_out, _ = self.gcn_bilstm(word_representation)  # BiLSTM applied to the word representation and used as Feature matrix for GCN, in order to produce a context-aware input for GCN
            matrix_adj = batch_adj_matrix(x_dependency_heads_matrix).to(self.device)  # following the suggestion of the GCN creator, in order to support the batch training I build a matrix with all adj matrices in his diagonal
            rel_emb = gcn_bilstm_out.reshape(batch_size * sequence_length, self.real_lstm_hidden_dim)
            out_gcn = self.gcn_layer(rel_emb, matrix_adj)
            out_gcn = out_gcn.view(batch_size, sequence_length, -1)
            word_representation = torch.cat((word_representation, out_gcn), dim=2)

        lstm_output, _ = self.lstm_emb(word_representation)

        if self.use_biaffine_layer:
            if self.use_predicate_biaffine:
                # Use the encoding of the predicate embeddings in combination with the core word representation as input for the biaffine attention layer
                predicate_embeddings_encoding = self.biaffine_lstm_predicate(predicates_emb)
                return self.biaffine_scorer(self.output_layer(lstm_output), self.output_layer_2(predicate_embeddings_encoding))
            else:
                # Use the deeper encoding for the biaffine attention layer
                stacked_lstm_output, _ = self.bilstm_stacked(lstm_output)
                return self.biaffine_scorer(self.output_layer(lstm_output), self.output_layer_2(stacked_lstm_output))

        return self.output_layer(lstm_output)
    

'''
TRAINER
Code inspired from notebook 3 by SAPIENZA NLP GROUP
'''
class Trainer():

    def __init__(
            self,
            model: nn.Module,
            loss_function,
            optimizer,
            label_vocab: vocab,
            device: str,
            log_steps: int = 1_000,
            log_level: int = 2,
    ):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device

        self.label_vocab = label_vocab
        self.log_steps = log_steps
        self.log_level = log_level

    def train(self, train_dataloader: DataLoader,
              dev_dataloader: DataLoader,
              epochs: int = 1, plot_loss: bool = False):
        if plot_loss:
            loss_summary_train = []
            loss_summary_valid = []
            f1_validation_summary_valid = []
            f1_classification_summary_valid = []
            epochs_list = list(range(epochs))

        train_loss = 0.0
        for epoch in range(epochs):
            # Save the model and optimizer state in order to resume the train phase at the right epochs.
            torch.save(self.model.state_dict(), "./srl-model/test_model_" + str(epoch) + ".pth")
            torch.save(self.optimizer.state_dict(), "./srl-model/test_optimizer_" + str(epoch) + ".pth")

            if plot_loss:
                # save at each epoch the loss and f1 into a object.
                print(loss_summary_train)
                print(loss_summary_valid)
                print(f1_validation_summary_valid)
                print(f1_classification_summary_valid)
                epochs_list_tmp = range(len(f1_classification_summary_valid))
                dict_loss = {"loss_train": loss_summary_train,
                             "loss_valid": loss_summary_valid,
                             "f1_valid_identification": f1_validation_summary_valid,
                             "f1_valid_classification": f1_classification_summary_valid,
                             "epochs_list": epochs_list_tmp
                             }
                print(dict_loss)
                torch.save(dict_loss, "./srl-model/DICT_LOSS" + str(epoch) + ".pth")

            if self.log_level > 0:
                print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss = 0.0
            self.model.train()

            for step, sample in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                label: torch.LongTensor = sample['label']

                self.optimizer.zero_grad()

                output = self.model(sample)
                
                if self.model.use_crf:
                    mask = (label != self.label_vocab['<pad>'])
                    loss = self.model.crf(output, label, mask=mask) * -1
                else:
                    output = output.view(-1, output.shape[-1])
                    label = label.view(-1)
                    loss = self.loss_function(output, label)

                #  backpropagation step
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.tolist()

                if self.log_level > 1 and step % self.log_steps == self.log_steps - 1:
                    # print intermediate average train loss
                    mid_loss = epoch_loss / (step + 1)
                    print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'.format(epoch, step, mid_loss))

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            train_loss += avg_epoch_loss
            if plot_loss:
                loss_summary_train.append(avg_epoch_loss)
                print(loss_summary_train)
            if self.log_level > 0:
                print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))  # print train loss at the end of the epoch

            if dev_dataloader is not None:
                if self.label_vocab is not None:
                    # at each epoch show the dev f1 score
                    result = print_output(self.model, dev_dataloader, self.label_vocab)
                    identification_valid_result = evaluate_argument_identification(dev_dataloader.dataset.labels, result)
                    f1_validation_summary_valid.append(identification_valid_result['f1'])  # append the identification dev f1 score, in order to plot it then
                    print("DEV IDENTIFICATION: ", identification_valid_result)
                    classification_valid_result = evaluate_argument_classification(dev_dataloader.dataset.labels, result)
                    f1_classification_summary_valid.append(classification_valid_result['f1'])  # append the identification dev f1 score, in order to plot it then
                    print("DEV CLASSIFICATION: ", classification_valid_result)
                valid_loss = self.evaluate(dev_dataloader)

                if plot_loss:
                    loss_summary_valid.append(valid_loss)
                    print(loss_summary_valid)

                print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch, valid_loss))

        avg_epoch_loss = train_loss / epochs
        print(loss_summary_train)
        print(loss_summary_valid)
        print(f1_validation_summary_valid)
        print(f1_classification_summary_valid)
        dict_loss = {"loss_train": loss_summary_train,
                     "loss_valid": loss_summary_valid,
                     "f1_valid_identification": f1_validation_summary_valid,
                     "f1_valid_classification": f1_classification_summary_valid,
                     "epochs_list": epochs_list
                     }
        print(dict_loss)
        torch.save(dict_loss, "./srl-model/DICT_LOSS" + str(epoch) + ".pth")
        return (avg_epoch_loss, dict_loss)

    def evaluate(self, valid_dataset):
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for sample in valid_dataset:
                labels = sample['label']
                predictions = self.model(sample)
                
                if self.model.use_crf:
                    mask = (labels != self.label_vocab['<pad>'])
                    loss = -1 * self.model.crf(predictions, labels, mask=mask)
                else:
                    labels = labels.view(-1)
                    predictions = predictions.view(-1, predictions.shape[-1])
                    loss = self.loss_function(predictions, labels)
                valid_loss += loss.tolist()
        return valid_loss / len(valid_dataset)



if __name__ == '__main__':

    print("Starting the training of the SRL model")
        
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DEVICE: str = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"

    # Some path for the training phase
    DATASET_PATH: str = './data/en_CANOPIES_100_train.json' 
    DATASET_DEV_PATH: str = './data/en_CANOPIES_10_dev.json' 
    GLOVE_PATH: str = "./glove-model/glove.6B.300d.txt"  # pre-trained glove embeddings path

    # read the dataset
    sentences, labels = read_dataset(DATASET_PATH)
    sentences_dev, labels_dev = read_dataset(DATASET_DEV_PATH)

    # -- Initialize bert --
    bert_config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name, config=bert_config)

    # -- net configuration -- It improve the code modularity
    net_configuration = net_configurator(use_bert_embeddings=USE_BERT_EMBEDDINGS, use_crf=USE_CRF, use_biaffine_layer=USE_BIAFFINE_LAYER, use_pretrained=USE_GLOVE, use_dependecy_heads=USE_DEPENDENCY_HEADS, use_predicates=False, use_syntagnet=USE_SYNTAGNET)

    dataset_train: SRL_Dataset = SRL_Dataset(sentences, labels, device=DEVICE, configurator=net_configuration, bert_model=bert_model, bert_tokenizer=bert_tokenizer)

    # creation of vocab starting from the train dataset
    vocab_words: vocab = build_vocab(dataset_train.sentences, 'words', min_freq=MIN_FREQUENCY)
    vocab_pos_tags: vocab = build_vocab(dataset_train.sentences, 'pos_tags')
    vocab_lemmas: vocab = build_vocab(dataset_train.sentences, 'lemmas', min_freq=MIN_FREQUENCY)
    vocab_predicates: vocab = build_vocab(dataset_train.sentences, 'predicates')
    vocab_label: vocab = build_vocab_roles(dataset_train.labels)

    store_dataset_train: dict = {"vocab_words": vocab_words,
                                "vocab_pos_tags": vocab_pos_tags,
                                "vocab_lemmas": vocab_lemmas,
                                "vocab_predicates": vocab_predicates,
                                "vocab_label": vocab_label
                                }

    # calculate the train dataset and save it
    torch.save(store_dataset_train, "./srl-model/dict_vocabs.pth")  # Save a copy of all vocabs
    dataset_train.build_sample(vocab_words, vocab_pos_tags, vocab_lemmas, vocab_predicates, vocab_label) #, vocab_dependency_relations)  # feature encoding phase
    torch.save(dataset_train, "./srl-model/train_dataset_stored.pth")

    # -- Hyperparameters class --
    @dataclass
    class HParams:
        label_vocabulary: vocab = vocab_label
        vocab_size_words: int = len(vocab_words)
        lstm_hidden_dim: int = 300
        embedding_dim_words: int = 300
        embedding_dim_lemmas: int = 300
        embedding_dim_relations: int = 300
        embedding_dim_predicates: int = 400
        embedding_dim_pos: int = 300
        gcn_output_dim: int = 143
        gcn_dropout_probability: float = 0.5 
        gcn_hidden_dim: int = 250
        gcn_lstm_num_layers: int = 2
        bert_hidden_dim: int = bert_config.hidden_size
        bert_lstm_num_layers: int = 2
        num_classes: int = len(vocab_label)
        bidirectional: bool = True
        num_layers: int = 2
        dropout: float = 0.3 
        lstm_dropout: float = 0.3 
        biaffine_lstm_num_layers: int = 2
        vocab_size_pos_tags: int = len(vocab_pos_tags)
        vocab_size_lemmas: int = len(vocab_lemmas)
        vocab_size_predicates: int = len(vocab_predicates)
        device: str = DEVICE


    hyperparameters: HParams = HParams()

    dataloader_train: DataLoader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    # Creation of the dev dataset
    dataset_dev: SRL_Dataset = SRL_Dataset(sentences_dev, labels_dev, device=DEVICE, max_len=dataset_train.max_len, configurator=net_configuration, bert_model=bert_model, bert_tokenizer=bert_tokenizer)
    dataset_dev.build_sample(vocab_words, vocab_pos_tags, vocab_lemmas, vocab_predicates, vocab_label) 
    dataloader_dev: DataLoader = DataLoader(dataset_dev, batch_size=BATCH_SIZE_VALID_TEST)

    if USE_GLOVE:
        '''
        Load the GloVe embeddings
        '''
        vectors = Vectors(GLOVE_PATH, cache="./")
        pretrained_embeddings = torch.randn(len(vocab_words), vectors.dim)
        initialised = 0
        for i, w in enumerate(vocab_words.get_itos()):
            if w in vectors.stoi:
                initialised += 1
                vec = vectors.get_vecs_by_tokens(w)
                pretrained_embeddings[i] = vec

        pretrained_embeddings[vocab_words[pad_token]] = torch.zeros(vectors.dim)
        hyperparameters.embedding_dim = vectors.dim
        hyperparameters.glove_embeddings = pretrained_embeddings
        hyperparameters.vocab_size_words = len(vocab_words)
        print("VECTOR DIM", vectors.dim)
        print("initialised embeddings {}".format(initialised))
        print("random initialised embeddings {} ".format(len(vocab_words) - initialised))

    print(hyperparameters)
    print(net_configuration)

    model: SRL_final_MODEL = SRL_final_MODEL(hparams=hyperparameters, configurator=net_configuration).to(DEVICE)
    trainer = Trainer(
        model=model,
        loss_function=torch.nn.CrossEntropyLoss(ignore_index=vocab_label[pad_token]),
        optimizer=torch.optim.Adam(model.parameters()),
        label_vocab=vocab_label,
        device=DEVICE,
    )


    _, dict_loss = trainer.train(dataloader_train, dataloader_dev, epochs=EPOCHS, plot_loss=PLOT_LOSS)
    plot_loss(dict_loss)  # plot the train and dev loss
    # store the model
    torch.save(model.state_dict(), "./srl-model/model_stored.pth")
    torch.save(trainer.optimizer.state_dict(), "./srl-model/model_stored_optimizer.pth")
    print("Model saved!")


    # DEV evaluation
    dev_predictions: dict = print_output(model, dataloader_dev, vocab_label)
    dev_arg_identification = evaluate_argument_identification(labels_dev, dev_predictions)
    print("DEV: IDENTIFICATION -> ", dev_arg_identification)
    dev_arg_classification = evaluate_argument_classification(labels_dev, dev_predictions)
    print("DEV: CLASSIFICATION -> ", dev_arg_classification)
    calculate_confusion_matrix(labels_dev, dev_predictions, 'DEV')  # plot the normalized confusion matrix
        

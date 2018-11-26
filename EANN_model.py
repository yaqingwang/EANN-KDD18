import numpy as np
import argparse
import time
# import random
import process_data_weibo as process_data
import copy
import cPickle as pickle
from random import sample
import torchvision
from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import torchvision.datasets as dsets
import torchvision.transforms as transforms

# from logger import Logger

from sklearn import metrics
from sklearn.preprocessing import label_binarize
import scipy.io as sio


class Rumor_Data(Dataset):
    # Custom Data Class
    def __init__(self, dataset):
        self.text = torch.from_numpy(np.array(dataset['post_text']))
        self.image = list(dataset['image'])  # torch.tensor
        self.mask = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        print('TEXT: %d, Image: %d, labe: %d, Event: %d'
              % (len(self.text), len(self.image), len(self.label), len(self.event_label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx], self.mask[idx]), self.label[idx], self.event_label[idx]


class ReverseLayerF(Function):
    # @staticmethod
    def forward(self, x):
        self.lambd = args.lambd
        return x.view_as(x)

    # @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x):
    return ReverseLayerF()(x)


# Neural Network Model (1 hidden layer)
class EANN(nn.Module):
    def __init__(self, args, W):
        super(EANN, self).__init__()
        self.args = args

        self.event_num = args.event_num

        vocab_size = args.vocab_size
        emb_dim = args.embed_dim

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19

        ### TEXT CNN
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(W))
        channel_in = 1
        filter_num = 20
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, emb_dim)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

        # Image
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)
        self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        ## Fake News Dectector
        self.news_detector = nn.Sequential()
        self.news_detector.add_module('c_fc1', nn.Linear(2 * self.hidden_size, 2))
        self.news_detector.add_module('c_softmax', nn.Softmax(dim=1))

        ###Event Discriminator
        self.event_discriminator = nn.Sequential()
        self.event_discriminator.add_module('d_fc1', nn.Linear(2 * self.hidden_size, self.hidden_size))
        self.event_discriminator.add_module('d_relu1', nn.LeakyReLU(True))
        self.event_discriminator.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.event_discriminator.add_module('d_softmax', nn.Softmax(dim=1))

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, text, image, mask):
        ### IMAGE #####
        image = self.vgg(image)  # [N, 512]
        image = F.leaky_relu(self.image_fc1(image))

        ##########Text CNN##################
        text = self.embed(text)
        text = text * mask.unsqueeze(2).expand_as(text)
        text = text.unsqueeze(1)
        text = [F.leaky_relu(conv(text)).squeeze(3) for conv in self.convs]  # [(N,hidden_dim,W), ...]*len(window_size)
        # text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,hidden_dim), ...]*len(window_size)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        text = F.leaky_relu(self.fc1(text))

        text_image = torch.cat((text, image), 1)

        ### Fake News Detection
        class_output = self.news_detector(text_image)
        ## Event Dscrimination
        reverse_feature = grad_reverse(text_image)
        event_output = self.event_discriminator(reverse_feature)

        return class_output, event_output


def to_var(x):
    try:
        if torch.cuda.is_available():
            x = x.cuda()
    except:
        print(x)
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def select(train, selec_indices):
    temp = []
    for i in range(len(train)):
        print("length is " + str(len(train[i])))
        print(i)
        # print(train[i])
        ele = list(train[i])
        temp.append([ele[i] for i in selec_indices])
    return temp


def split_train_validation(train, percent):
    whole_len = len(train[0])

    train_indices = (sample(range(whole_len), int(whole_len * percent)))
    train_data = select(train, train_indices)
    print("train data size is " + str(len(train[3])))
    # print()

    validation = select(train, np.delete(range(len(train[0])), train_indices))
    print("validation size is " + str(len(validation[3])))
    print("train and validation data set has been splited")

    return train_data, validation


def main(args):
    print('loading data')

    # Dataset
    train, validation, test, W = load_data(args)
    test_id = test['post_id']

    train_dataset = Rumor_Data(train)

    validate_dataset = Rumor_Data(validation)

    test_dataset = Rumor_Data(test)

    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    print('building model')
    model = EANN(args, W)

    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 lr=args.learning_rate)

    iter_per_epoch = len(train_loader)
    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_test_acc = 0.000
    best_loss = 100
    best_validate_dir = ''
    best_list = [0, 0]

    print('training model')
    # Train the Model
    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        lr = 0.001 / (1. + 10 * p) ** 0.75
        optimizer.lr = lr
        start_time = time.time()
        cost_vector = []
        news_cost_vector = []
        event_cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        test_acc_vector = []
        vali_cost_vector = []
        test_cost_vector = []

        for i, (train_data, train_labels, event_labels) in enumerate(train_loader):
            train_text, train_image, train_mask, train_labels, event_labels = \
                to_var(train_data[0]), to_var(train_data[1]), to_var(train_data[2]), \
                to_var(train_labels), to_var(event_labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            news_outputs, event_outputs = model(train_text, train_image, train_mask)
            news_loss = criterion(news_outputs, train_labels)
            event_loss = criterion(event_outputs, event_labels)
            loss = news_loss + event_loss
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(news_outputs, 1)

            #cross_entropy = True

            if True:
                accuracy = (train_labels == argmax.squeeze()).float().mean()
            else:
                _, labels = torch.max(train_labels, 1)
                accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()

            news_cost_vector.append(news_loss.data[0])
            event_cost_vector.append(event_loss.data[0])
            cost_vector.append(loss.data[0])
            acc_vector.append(accuracy.data[0])

        model.eval()
        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels, event_labels) in enumerate(validate_loader):
            validate_text, validate_image, validate_mask, validate_labels, event_labels = \
                to_var(validate_data[0]), to_var(validate_data[1]), to_var(validate_data[2]), \
                to_var(validate_labels), to_var(event_labels)
            validate_outputs, domain_outputs = model(validate_text, validate_image, validate_mask)
            _, validate_argmax = torch.max(validate_outputs, 1)
            vali_loss = criterion(validate_outputs, validate_labels)
            validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append(vali_loss.data[0])
            validate_acc_vector_temp.append(validate_accuracy.data[0])
        validate_acc = np.mean(validate_acc_vector_temp)
        valid_acc_vector.append(validate_acc)
        model.train()

        print ('Epoch [%d/%d],  Loss: %.4f, News Loss: %.4f,  Event loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
               % (
                   epoch + 1, args.num_epochs, np.mean(cost_vector), np.mean(news_cost_vector),
                   np.mean(event_cost_vector),
                   np.mean(acc_vector), validate_acc))

        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            best_validate_dir = args.output_file + str(epoch + 1) + '.pkl'
            best_list[0] = best_validate_dir
            torch.save(model.state_dict(), best_validate_dir)
        duration = time.time() - start_time

    # Test the Model
    print('testing model')
    model = EANN(args, W)
    model.load_state_dict(torch.load(best_validate_dir))
    #    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
        test_text, test_image, test_mask, test_labels = to_var(
            test_data[0]), to_var(test_data[1]), to_var(test_data[2]), to_var(test_labels)
        test_outputs, _ = model(test_text, test_image, test_mask)
        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = to_np(test_outputs.squeeze())
            test_pred = to_np(test_argmax.squeeze())
            test_true = to_np(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs.squeeze())), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='micro')
    test_precision = metrics.precision_score(test_true, test_pred, average='micro')
    test_recall = metrics.recall_score(test_true, test_pred, average='micro')

    test_true_convert = label_binarize(test_true, classes=range(2))
    test_aucroc = metrics.roc_auc_score(test_true_convert, test_score.T[0], average='micro')
    test_aucpr = metrics.average_precision_score(test_true_convert, test_score.T[0], average='micro')
    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    print("Classification Acc: %.4f, AUC-ROC: %.4f, AUC-PR: %.4f"
          % (test_accuracy, test_aucroc, test_aucpr))
    print("Classification report:\n%s\n"
          % (metrics.classification_report(test_true, test_pred)))
    print("Classification confusion matrix:\n%s\n"
          % (test_confusion_matrix))

    print('Saving results')
    obj_arr = np.zeros((6,), dtype=np.object)
    obj_arr[0] = args
    obj_arr[1] = [test_aucroc, test_aucpr, test_precision, test_recall, test_f1, test_accuracy]
    obj_arr[2] = test_confusion_matrix
    obj_arr[3] = test_score
    obj_arr[4] = test_pred
    obj_arr[5] = test_true
    sio.savemat(best_validate_dir + '_results.mat', {'results': obj_arr})


def parse_arguments(parser):
    parser.add_argument('data_file', type=str, metavar='<data_file>', help='')
    parser.add_argument('word_embedding_file', type=str, metavar='<word_embedding_file>', help='')
    parser.add_argument('output_file', type=str, metavar='<output_file>', help='')

    parse.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=100, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')
    return parser


def word2vec(post, word_id_map, W):
    word_embedding = []
    mask = []
    for sentence in post:
        sen_embedding = []
        seq_len = len(sentence) - 1
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word_id_map[word])

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)
        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
    return word_embedding, mask


def load_data(args):
    #data = pickle.load(open(args.data_file, 'rb'))
    data = process_data.main()
    train, validate, test = data[0], data[1], data[2]
    word_vector_path = args.word_embedding_file
    f = open(word_vector_path, 'rb')
    weight = pickle.load(f)  # W: pretrained_word_embdding, W2: Random initiliazed embedding, word_idx_map, vocab
    W, W2, word_idx_map, vocab, max_len = weight[0], weight[1], weight[2], weight[3], weight[4]

    args.vocab_size = len(vocab)
    args.sequence_len = max_len
    print("translate data to embedding")

    word_embedding, mask = word2vec(validate['post_text'], word_idx_map, W)
    validate['post_text'] = word_embedding
    validate['mask'] = mask

    print("translate test data to embedding")
    word_embedding, mask = word2vec(test['post_text'], word_idx_map, W)
    test['post_text'] = word_embedding
    test['mask'] = mask
    word_embedding, mask = word2vec(train['post_text'], word_idx_map, W)
    train['post_text'] = word_embedding
    train['mask'] = mask
    print("sequence length " + str(args.sequence_length))
    print("Train Data Size is " + str(len(train['post_text'])))
    print("Finished loading data ")
    return train, validate, test, W


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    weibo_data = '../Data/weibo/weibo_data.pickle'
    word_vector_path = '../Data/weibo/word_embedding.pickle'
    output = '../Data/weibo/result/'

    args = parser.parse_args([weibo_data, word_vector_path, output])
    main(args)




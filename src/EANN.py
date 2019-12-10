import numpy as np
import argparse
import time, os
# import random
import process_data_weibo as process_data
import copy
import pickle as pickle
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

#from logger import Logger

from sklearn import metrics
from sklearn.preprocessing import label_binarize
import scipy.io as sio

class Rumor_Data(Dataset):
    def __init__(self, dataset):
        self.text = torch.from_numpy(np.array(dataset['post_text']))
        self.image = list(dataset['image'])
        #self.social_context = torch.from_numpy(np.array(dataset['social_feature']))
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

    #@staticmethod
    def forward(self, x):
        self.lambd = args.lambd
        return x.view_as(x)

    #@staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x):
    return ReverseLayerF()(x)



# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, args, W):
        super(CNN_Fusion, self).__init__()
        self.args = args

        self.event_num = args.event_num

        vocab_size = args.vocab_size
        emb_dim = args.embed_dim

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19

        # TEXT RNN
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(W))
        self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        self.text_encoder = nn.Linear(emb_dim, self.hidden_size)

        ### TEXT CNN
        channel_in = 1
        filter_num = 20
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, emb_dim)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)

        self.dropout = nn.Dropout(args.dropout)

        #IMAGE
        #hidden_size = args.hidden_dim
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs,  self.hidden_size)
        #self.image_fc2 = nn.Linear(512, self.hidden_size)
        self.image_adv = nn.Linear(self.hidden_size,  int(self.hidden_size))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        ###social context
        self.social = nn.Linear(self.social_size, self.hidden_size)

        ##ATTENTION
        self.attention_layer = nn.Linear(self.hidden_size, emb_dim)

        ## Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(2 * self.hidden_size, 2))
        #self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        #self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        #self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        #self.class_classifier.add_module('c_fc2', nn.Linear(self.hidden_size, 2))
        #self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(self.hidden_size))
        #self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        #self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        ###Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(2 * self.hidden_size, self.hidden_size))
        #self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        #x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, text, image,  mask):
        ### IMAGE #####
        image = self.vgg(image) #[N, 512]
        image = F.leaky_relu(self.image_fc1(image))
        
        ##########CNN##################
        text = self.embed(text)
        text = text * mask.unsqueeze(2).expand_as(text)
        text = text.unsqueeze(1)
        text = [F.leaky_relu(conv(text)).squeeze(3) for conv in self.convs]  # [(N,hidden_dim,W), ...]*len(window_size)
        #text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,hidden_dim), ...]*len(window_size)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        text = F.leaky_relu(self.fc1(text))
        text_image = torch.cat((text, image), 1)

        ### Fake or real
        class_output = self.class_classifier(text_image)
        ## Domain (which Event )
        reverse_feature = grad_reverse(text_image)
        domain_output = self.domain_classifier(reverse_feature)

        # ### Multimodal
        # text_reverse_feature = grad_reverse(text)
        # image_reverse_feature = grad_reverse(image)
        # text_output = self.modal_classifier(text_reverse_feature)
        # image_output = self.modal_classifier(image_reverse_feature
        return class_output, domain_output

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()

def select(train, selec_indices):
    temp = []
    for i in range(len(train)):
        print("length is "+str(len(train[i])))
        print(i)
        #print(train[i])
        ele = list(train[i])
        temp.append([ele[i] for i in selec_indices])
    return temp

def make_weights_for_balanced_classes(event, nclasses = 15):
    count = [0] * nclasses
    for item in event:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(event)
    for idx, val in enumerate(event):
        weight[idx] = weight_per_class[val]
    return weight

def split_train_validation(train, percent):
    whole_len = len(train[0])

    train_indices = (sample(range(whole_len), int(whole_len * percent)))
    train_data = select(train, train_indices)
    print("train data size is "+ str(len(train[3])))
    # print()

    validation = select(train, np.delete(range(len(train[0])), train_indices))
    print("validation size is "+ str(len(validation[3])))
    print("train and validation data set has been splited")

    return train_data, validation


def main(args):
    print('loading data')
    #    dataset = DiabetesDataset(root=args.training_file)
    #    train_loader = DataLoader(dataset=dataset,
    #                              batch_size=32,
    #                              shuffle=True,
    #                              num_workers=2)

    # MNIST Dataset
    train, validation, test, W = load_data(args)
    test_id = test['post_id']

    #train, validation = split_train_validation(train,  1)

    #weights = make_weights_for_balanced_classes(train[-1], 15)
    #weights = torch.DoubleTensor(weights)
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))


    train_dataset = Rumor_Data(train)

    validate_dataset = Rumor_Data(validation)

    test_dataset = Rumor_Data(test) 

    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    validate_loader = DataLoader(dataset = validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    print('building model')
    model = CNN_Fusion(args, W)

    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 lr= args.learning_rate)
    #optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 #lr=args.learning_rate)
    #scheduler = StepLR(optimizer, step_size= 10, gamma= 1)


    iter_per_epoch = len(train_loader)
    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_test_acc = 0.000
    best_loss = 100
    best_validate_dir = ''
    best_list = [0,0]

    print('training model')
    adversarial = True
    # Train the Model
    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        #lambd = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.001 / (1. + 10 * p) ** 0.75

        optimizer.lr = lr
        #rgs.lambd = lambd
        start_time = time.time()
        cost_vector = []
        class_cost_vector = []
        domain_cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        test_acc_vector = []
        vali_cost_vector = []
        test_cost_vector = []

        for i, (train_data, train_labels, event_labels) in enumerate(train_loader):
            train_text, train_image,  train_mask, train_labels, event_labels = \
                to_var(train_data[0]), to_var(train_data[1]), to_var(train_data[2]), \
                to_var(train_labels), to_var(event_labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()

            class_outputs, domain_outputs = model(train_text, train_image, train_mask)

            ## Fake or Real loss
            class_loss = criterion(class_outputs, train_labels)
            # Event Loss
            domain_loss = criterion(domain_outputs, event_labels)
            loss = class_loss + domain_loss
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(class_outputs, 1)

            cross_entropy = True

            if True:
                accuracy = (train_labels == argmax.squeeze()).float().mean()
            else:
                _, labels = torch.max(train_labels, 1)
                accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()

            class_cost_vector.append(class_loss.data[0])
            domain_cost_vector.append(domain_loss.data[0])
            cost_vector.append(loss.data[0])
            acc_vector.append(accuracy.data[0])
            # if i == 0:
            #     train_score = to_np(class_outputs.squeeze())
            #     train_pred = to_np(argmax.squeeze())
            #     train_true = to_np(train_labels.squeeze())
            # else:
            #     class_score = np.concatenate((train_score, to_np(class_outputs.squeeze())), axis=0)
            #     train_pred = np.concatenate((train_pred, to_np(argmax.squeeze())), axis=0)
            #     train_true = np.concatenate((train_true, to_np(train_labels.squeeze())), axis=0)



        model.eval()
        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels, event_labels) in enumerate(validate_loader):
            validate_text, validate_image,  validate_mask, validate_labels, event_labels = \
                to_var(validate_data[0]), to_var(validate_data[1]), to_var(validate_data[2]), \
                to_var(validate_labels), to_var(event_labels)
            validate_outputs, domain_outputs = model(validate_text, validate_image, validate_mask)
            _, validate_argmax = torch.max(validate_outputs, 1)
            vali_loss = criterion(validate_outputs, validate_labels)
            #domain_loss = criterion(domain_outputs, event_labels)
                #_, labels = torch.max(validate_labels, 1)
            validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append( vali_loss.data[0])
                #validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            validate_acc_vector_temp.append(validate_accuracy.data[0])
        validate_acc = np.mean(validate_acc_vector_temp)
        valid_acc_vector.append(validate_acc)
        model.train()
        print ('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, domain loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
                % (
                epoch + 1, args.num_epochs,  np.mean(cost_vector), np.mean(class_cost_vector),  np.mean(domain_cost_vector),
                    np.mean(acc_vector),   validate_acc))

        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            if not os.path.exists(args.output_file):
                os.mkdir(args.output_file)

            best_validate_dir = args.output_file + str(epoch + 1) + '.pkl'
            torch.save(model.state_dict(), best_validate_dir)

        duration = time.time() - start_time
        # print ('Epoch: %d, Mean_Cost: %.4f, Duration: %.4f, Mean_Train_Acc: %.4f, Mean_Test_Acc: %.4f'
        # % (epoch + 1, np.mean(cost_vector), duration, np.mean(acc_vector), np.mean(test_acc_vector)))
        # best_validate_dir = args.output_file + 'weibo_GPU2_out.' + str(52) + '.pkl'
    


    # Test the Model
    print('testing model')
    model = CNN_Fusion(args, W)
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
        test_outputs, domain_outputs= model(test_text, test_image, test_mask)
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
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')
    
    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    print("Classification Acc: %.4f, AUC-ROC: %.4f"
          % (test_accuracy, test_aucroc))
    print("Classification report:\n%s\n"
          % (metrics.classification_report(test_true, test_pred)))
    print("Classification confusion matrix:\n%s\n"
          % (test_confusion_matrix))



def parse_arguments(parser):
    parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    #parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    parser.add_argument('output_file', type=str, metavar='<output_file>', help='')

    parse.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default = 32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default= 1, help='')
    parser.add_argument('--text_only', type=bool, default= False, help='')

    #    parser.add_argument('--sequence_length', type = int, default = 28, help = '')
    #    parser.add_argument('--input_size', type = int, default = 28, help = '')
    #    parser.add_argument('--hidden_size', type = int, default = 128, help = '')
    #    parser.add_argument('--num_layers', type = int, default = 2, help = '')
    #    parser.add_argument('--num_classes', type = int, default = 10, help = '')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=100, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')

    #    args = parser.parse_args()
    return parser


def get_top_post(output, label, test_id, top_n = 500):
    filter_output = []
    filter_id = []
    #print(test_id)
    #print(output)
    for i, l in enumerate(label):
        #print(np.argmax(output[i]))
        if np.argmax(output[i]) == l and int(l) == 1 :
            filter_output.append(output[i][1])
            filter_id.append(test_id[i])

    filter_output = np.array(filter_output)

    top_n_indice = filter_output.argsort()[-top_n:][::-1]

    top_n_id = np.array(filter_id)[top_n_indice]
    top_n_id_dict = {}
    for i in top_n_id:
        top_n_id_dict[i] = True

    pickle.dump(top_n_id_dict, open("../Data/weibo/top_n_id.pickle", "wb"))

    return top_n_id





def word2vec(post, word_id_map, W):
    word_embedding = []
    mask = []
    #length = []

    for sentence in post:
        sen_embedding = []
        seq_len = len(sentence) -1
        mask_seq = np.zeros(args.sequence_len, dtype = np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word_id_map[word])

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)


        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
        #length.append(seq_len)
    return word_embedding, mask

def load_data(args):
    train, validate, test = process_data.get_data(args.text_only)
    #print(train[4][0])
    word_vector_path = '../Data/weibo/word_embedding.pickle'
    f = open(word_vector_path, 'rb')
    weight = pickle.load(f)  # W, W2, word_idx_map, vocab
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
    test['mask']=mask
    #test[-2]= transform(test[-2])
    word_embedding, mask = word2vec(train['post_text'], word_idx_map, W)
    train['post_text'] = word_embedding
    train['mask'] = mask
    print("sequence length " + str(args.sequence_length))
    print("Train Data Size is "+str(len(train['post_text'])))
    print("Finished loading data ")
    return train, validate, test, W

def transform(event):
    matrix = np.zeros([len(event), max(event) + 1])
    #print("Translate  shape is " + str(matrix))
    for i, l in enumerate(event):
        matrix[i, l] = 1.00
    return matrix

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    train = '' 
    test = ''
    output = '../Data/weibo/RESULT/'
    args = parser.parse_args([train, test, output])
    
    main(args)
   


import numpy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle
import ffnn as FFNN

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()
        self.numOfLayer = 3
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=False)
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W1 = nn.Linear(input_dim,h)
        self.W2 = nn.Linear(h,32)
        self.W3 = nn.Linear(32, 4)
        self.softmax = nn.LogSoftmax(dim=2)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # First hidden layer with activation and optional dropout
        hiddenLayer1 = self.activation(self.W1(input_vector))
        hiddenLayer1 = self.dropout(hiddenLayer1)  # Dropout layer if defined in __init__

        # Second hidden layer (if you add another layer in __init__)
        hiddenLayer2 = self.activation(self.W2(hiddenLayer1))
        hiddenLayer2 = self.dropout(hiddenLayer2)

        # Output layer (logits before softmax)
        outputVec = self.W3(hiddenLayer2)

        # Probability distribution with softmax
        predictedVector = self.softmax(outputVec)
        return predictedVector



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    # Step 1: Load the word embeddings from the pickle file
    with open('./word_embedding.pkl', 'rb') as f:
        word_embedding = pickle.load(f)

    # Step 2: Convert word embeddings dictionary to tensor
    # Create word-to-index and index-to-embedding mappings
    word_to_index = {word: i for i, word in enumerate(word_embedding.keys())}
    index_to_embedding = [torch.tensor(word_embedding[word], dtype=torch.float32) for word in word_embedding.keys()]

    # Stack embeddings into a single tensor for use in nn.Embedding
    embedding_tensor = torch.stack(index_to_embedding)
    input_dim = embedding_tensor.shape[1]

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this
    print(embedding_tensor.shape)
    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)  # Fill in parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # word_embedding = pickle.load(open('word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0
    best_train_accuracy = 0
    best_validation_accuracy = 0
    # model = FFNN(input_dim, h = args.hidden_dim)
    # optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]

                # Transform the input into required shape
                vectors = numpy.array(vectors)
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        print(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        training_accuracy = correct/total


        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(training_accuracy))
        valid_data = valid_data

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words]

            vectors = numpy.array(vectors)
            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
            # print(predicted_label, gold_label)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        validation_accuracy = correct/total

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
        if training_accuracy > best_train_accuracy:
            best_train_accuracy = training_accuracy
        if validation_accuracy < (last_validation_accuracy - 0.02) and training_accuracy > (last_train_accuracy + 0.02):
            stopping_condition=True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = training_accuracy
        if epoch == args.epochs:
            stopping_condition = True

        epoch += 1



    # You may find it beneficial to keep track of training accuracy or training loss;
    print("Best Accuracies: Training-", best_train_accuracy, " Validation-", best_validation_accuracy)
    print("Last Epoch Accuracies: Training-", last_train_accuracy, " Validation-",last_validation_accuracy)
    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance

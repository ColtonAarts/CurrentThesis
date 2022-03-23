import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TextClassifier(nn.ModuleList):

    def __init__(self, params):
        super(TextClassifier, self).__init__()

        # Parameters regarding text preprocessing
        self.seq_len = params.seq_len
        self.num_words = params.num_words
        self.embedding_size = params.embedding_size

        self.conv_list = None
        
        # Dropout definition
        self.dropout = nn.Dropout(0.25)
        
        # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = 3
        
        # Output size for each convolution
        self.out_size = params.out_size
        # Number of strides for each convolution
        self.stride = params.stride
        
        # Embedding layer definition
        self.embedding = nn.Embedding(self.num_words + 1, self.embedding_size, padding_idx=0)
        
        # Convolution layers definition
        self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)


        
        # Max pooling layers definition
        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)

        self.flatten = nn.Flatten()
        # print(math.floor((self.embedding_size/self.stride/self.stride) - 4*self.stride))
        self.biLstm = nn.LSTM(math.floor((self.embedding_size/self.stride/self.stride) - 4*self.stride), 100, 1, bidirectional=True, batch_first=True)

        # Fully connected layer definition
        self.fc = nn.Linear(49600, 4)

    def forward(self, x):
        # Sequence of tokes is filterd through an embedding layer
        # print(f" dimensions: {x.size()}")
        x = self.embedding(x)
        # print(f"Embedding dimensions: {x.size()}")
        # Convolution layer 1 is applied
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        # print(f"Conv Dimenstions: {x1.size()}")
        x1 = self.pool_1(x1)
        # print(f"Pooling dimesions: {x1.size()}")

        x1 = self.biLstm(x1)
        # print(f"LSTM Dim: {x1[0].size()}")
        x1 = torch.tanh(x1[0])

        x1 = self.flatten(x1)
        if self.conv_list is not None:
            for ele in x1.tolist():
                self.conv_list.append(ele)
        # print(f"Dim to Dense: {x1.size()}")

        # out = self.dropout(x1)

        # The "flattened" vector is passed through a fully connected layer
        out = self.fc(x1)

        # print(f"Out size: {out.size()}")

        # # print(out)

        # Dropout is applied        

        # Activation function is applied
        # out = torch.sigmoid(out)

        # out = torch.softmax(out, 4)


        # out = torch.nn.CrossEntropyLoss(out)
        return out

    def convolution(self, x):
        # Sequence of tokes is filterd through an embedding layer
        x = self.embedding(x)
        # print(f"Embedding dimensions: {x.size()}")
        # Convolution layer 1 is applied
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        # print(f"Conv Dimenstions: {x1.size()}")
        x1 = self.pool_1(x1)
        # print(f"Pooling dimesions: {x1.size()}")

        x1 = self.biLstm(x1)
        # print(f"LSTM Dim: {x1[0].size()}")
        x1 = torch.tanh(x1[0])

        x1 = self.flatten(x1)
        return x1


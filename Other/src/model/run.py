import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report

from torch.utils.data import Dataset, DataLoader

class DatasetMaper(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Run:
    '''Training, evaluation and metrics calculation'''

    @staticmethod
    def train(model, data, params):
        
        # Initialize dataset maper
        train = DatasetMaper(data['x_train'], data['y_train'])
        test = DatasetMaper(data['x_test'], data['y_test'])
        
        # Initialize loaders
        loader_train = DataLoader(train, batch_size=params.batch_size)
        loader_test = DataLoader(test, batch_size=params.batch_size)
        
        # Define optimizer
        optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate)
        
        # Starts training phase
        for epoch in range(params.epochs):
            print(epoch)
            # Set model in training model
            model.train()
            predictions = []
            # Starts batch training
            for x_batch, y_batch in loader_train:
                # y_batch = y_batch.type(torch.FloatTensor)
                y_batch = torch.tensor(list(y_batch), dtype=torch.int64)
                # Feed the model
                x_batch = torch.tensor(x_batch).to(torch.int64)
                y_pred = model(x_batch)
                # Loss calculation
                #
                # print(x_batch)
                # print(len(x_batch))
                #
                # print(len(y_pred))
                # print(y_pred)
                # print(len(y_batch))
                # print(len(y_pred.detach().numpy()))
                #
                # results = list()
                # count = 0
                # for value in y_pred.detach().numpy():
                #
                #     max_val = np.max(value)
                #     # print(f"value:\n{value}")
                #     ind = np.where(value == max_val)
                #     results.append(ind[0][0])
                # print(results)
                # y_pred = torch.tensor(results, dtype=torch.int32)
                # #

                loss = torch.nn.CrossEntropyLoss()
                output = loss(y_pred, y_batch)
                
                # Clean gradientes
                optimizer.zero_grad()
                
                # Gradients calculation
                output.backward()
                
                # Gradients update
                optimizer.step()
                
                # Save predictions
                predictions += list(y_pred.detach().numpy())
            
            # Evaluation phase
        test_predictions = Run.evaluation(model, loader_test)

        # Metrics calculation
        print(test_predictions)
        results = list()
        for ele in test_predictions:
            ele = list(ele)
            max_ele = max(ele)
            index = ele.index(max_ele)
            results.append(index)
        print(classification_report(data['y_test'], results))
            # train_accuary = Run.calculate_accuray(data['y_train'], predictions)
            # test_accuracy = Run.calculate_accuray(data['y_test'], test_predictions)
            # print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch+1, loss.item(), train_accuary, test_accuracy))


            
    @staticmethod
    def evaluation(model, loader_test):
        
        # Set the model in evaluation mode
        model.eval()
        predictions = []
        
        # Starst evaluation phase
        with torch.no_grad():
            for x_batch, y_batch in loader_test:
                x_batch = torch.tensor(x_batch).to(torch.int64)
                y_pred = model(x_batch)
                predictions += list(y_pred.detach().numpy())

        return predictions
        
    @staticmethod
    def calculate_accuray(grand_truth, predictions):
        # Metrics calculation
        true_positives = 0
        true_negatives = 0
        for true, pred in zip(grand_truth, predictions):
            if (pred >= 0.5) and (true == 1):
                true_positives += 1
            elif (pred < 0.5) and (true == 0):
                true_negatives += 1
            else:
                pass
        # Return accuracy
        return (true_positives+true_negatives) / len(grand_truth)
        
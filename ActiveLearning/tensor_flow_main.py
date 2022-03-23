from Other.src import Parameters
from Other.src import Preprocessing
from Other.src import TextClassifier
from Other.src import Different_Preprocessing
from Other.src import Run
from torch.utils.data import DataLoader
from Other.src.model.run import DatasetMaper
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
import torch.optim as optim
from ActiveLearning.MaximumUncertainty import entropy as m_u_entropy
from ActiveLearning.MaximumUncertainty import max_similarity as m_u_max_similarity
from ActiveLearning.MaximumUncertainty import max_cosine_similarity as cosine_sim
import numpy as np
from scipy import spatial
from ActiveLearning import NeuralNetwork

import torch
import MyCLSTM
import MyPreprocessing


class Controller(Parameters):
    def __init__(self):
        # Preprocessing pipeline
        path_train = "C:\\Users\\aarts\\Documents\\CPSC 371\\EmbeddingTest2\\Other\\data\\training_25.csv"
        path_test = "C:\\Users\\aarts\\Documents\\CPSC 371\\EmbeddingTest2\\Other\\data\\testing_25.csv"
        self.data = Different_Preprocessing(Parameters.num_words, Parameters.seq_len, path_train, path_test)
        self.data.do_things()
        self.svd_matrix = self.create_svd()
        # Initialize the model

        self.model = NeuralNetwork.NeuralNetwork(Parameters.seq_len, 4)

        # self.model = MyCLSTM.CLSTM(248,1,250)
        # self.data = self.my_prepare_data(Parameters.num_words, Parameters.seq_len, False)
        # Training - Evaluation pipeline

    def create_svd(self):
        # starting_x_data is the data that the classifier will be given to start so it does not need to be included in
        # the svd representation
        starting_x_data, data, starting_y_data, remaining_y_data = train_test_split(self.data.x_after_split,
                                                                                    self.data.y_train,train_size=.05,
                                                                                    random_state=42)
        docs = list()
        for ele in data:
            s = ""
            for word in ele:
                s = s + " " + word
            docs.append(s)

        tf_idf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
        svd = TruncatedSVD(100)

        tf_idf_matrix = tf_idf.fit_transform(docs)
        return svd.fit_transform(tf_idf_matrix)

    def convert_results(self, results):
        final_results = list()

        for prediction in results:
            max_percent = max(prediction)
            indexes = prediction.tolist().index(max_percent)

            final_results.append(indexes)
        return final_results

    def train(self):
        out_file = open("results_active_learning2.csv", "w+")
        out_file.write("num,0,1,2,3\n")
        start_data_x,  rest_data_x, start_data_y, rest_data_y = train_test_split(self.data.x_train, self.data.y_train,
                                                                                 train_size=.05, random_state=42)

        # all_training_data_x = start_data_x.tolist()
        # all_training_data_y = start_data_y.tolist()

        all_training_data_x = list()
        all_training_data_y = list()

        for index, value in enumerate(start_data_x):
            all_training_data_x.append(value)
            all_training_data_y.append(start_data_y[index])
        # Starts training phase

        self.model.fit(start_data_x, start_data_y)
        results = self.convert_results(self.model.predict(self.data.x_test))

        report = classification_report(self.data.y_test, results, output_dict=True)
        zero = report["0"]["f1-score"]
        one = report["1"]["f1-score"]
        two = report["2"]["f1-score"]
        three = report["3"]["f1-score"]

        print(f"{len(start_data_x)},{zero},{one},{two},{three}")
        out_file.write(f"{len(start_data_x)},{zero},{one},{two},{three}\n")

        self.exploration_p(all_training_data_x, all_training_data_y, rest_data_x, rest_data_y, out_file)

    def add_to_selected_and_all(self, selected_data_x, selected_data_y, all_x, all_y, element_x, element_y):
        selected_data_x.append(np.asarray(element_x))
        selected_data_y.append(element_y)
        all_x.append(np.asarray(element_x))
        all_y.append(element_y)

    def exploration_p(self, all_training_data_x, all_training_data_y, rest_data_x, rest_data_y, out_file):
        average = 0

        while average < Parameters.target:

            test_predictions = self.model.predict(rest_data_x)
            entropy_list = list()
            for ele in test_predictions:
                ele = list(ele)
                ent = m_u_entropy(ele)
                entropy_list.append(ent)

            index_ent = entropy_list.index(max(entropy_list))
            del entropy_list[index_ent]
            selected_data_x = list()
            selected_data_y = list()

            ele_x = list()

            for ele in rest_data_x[index_ent]:
                ele_x.append(ele.item())
            ele_x = np.asarray(ele_x)
            ele_y = rest_data_y[index_ent]

            self.add_to_selected_and_all(selected_data_x, selected_data_y, all_training_data_x, all_training_data_y,
                                         ele_x, ele_y)

            rest_data_x = np.delete(rest_data_x, index_ent, 0)
            rest_data_y = np.delete(rest_data_y, index_ent, 0)

            selected_data_matrices = list()

            thing = self.svd_matrix[index_ent]
            self.svd_matrix = np.delete(self.svd_matrix, index_ent, 0)

            selected_data_matrices.append(thing)

            while len(selected_data_x) < Parameters.num_exploitation:

                max_sim_lst = list()

                for ele in self.svd_matrix:
                    max_sim = cosine_sim(ele, selected_data_matrices)
                    max_sim_lst.append(max_sim)

                ent_minus_sim_list = list()
                for count, value in enumerate(entropy_list):
                    ent_minus_sim_list.append(value - Parameters.alpha * max_sim_lst[count])

                index_to_add = ent_minus_sim_list.index(max(ent_minus_sim_list))

                ent_minus_sim_list.clear()

                self.add_to_selected_and_all(selected_data_x, selected_data_y, all_training_data_x, all_training_data_y,
                                             rest_data_x[index_to_add], rest_data_y[index_to_add])

                thing = self.svd_matrix[index_to_add]
                self.svd_matrix = np.delete(self.svd_matrix, index_to_add, 0)

                selected_data_matrices.append(thing)

                rest_data_x = np.delete(rest_data_x, index_to_add, 0)
                rest_data_y = np.delete(rest_data_y, index_to_add, 0)

                del entropy_list[index_to_add]

            # Exploration

            max_sim_lst = list()

            for ele in self.svd_matrix:
                max_sim = cosine_sim(ele, selected_data_matrices)
                max_sim_lst.append(max_sim)

            index_to_add = max_sim_lst.index(min(max_sim_lst))

            self.add_to_selected_and_all(selected_data_x, selected_data_y, all_training_data_x, all_training_data_y,
                                         rest_data_x[index_to_add], rest_data_y[index_to_add])

            thing = self.svd_matrix[index_to_add]
            self.svd_matrix = np.delete(self.svd_matrix, index_to_add, 0)

            selected_data_matrices.append(thing)

            rest_data_x = np.delete(rest_data_x, index_to_add, 0)
            rest_data_y = np.delete(rest_data_y, index_to_add, 0)
            del max_sim_lst[index_to_add]

            del entropy_list[index_to_add]

            while len(selected_data_x) < Parameters.total_num:
                for index, ele in enumerate(self.svd_matrix):
                    redun = 1 - spatial.distance.cosine(thing, ele)
                    if redun > max_sim_lst[index]:
                        max_sim_lst[index] = redun
                index_to_add = max_sim_lst.index(min(max_sim_lst))

                self.add_to_selected_and_all(selected_data_x, selected_data_y, all_training_data_x, all_training_data_y,
                                             rest_data_x[index_to_add], rest_data_y[index_to_add])
                thing = self.svd_matrix[index_to_add]
                self.svd_matrix = np.delete(self.svd_matrix, index_to_add, 0)

                selected_data_matrices.append(thing)

                rest_data_x = np.delete(rest_data_x, index_to_add, 0)
                rest_data_y = np.delete(rest_data_y, index_to_add, 0)
                del max_sim_lst[index_to_add]

                del entropy_list[index_to_add]

            max_sim_lst.clear()
            entropy_list.clear()
            selected_data_y.clear()
            selected_data_x.clear()
            selected_data_matrices.clear()

            print(f"length training: {len(all_training_data_x)}")
            print(len(all_training_data_x[0]))
            print(np.asarray(all_training_data_x).shape)

            self.model = NeuralNetwork.NeuralNetwork(Parameters.seq_len, 4)
            self.model.fit(np.array(all_training_data_x), np.array(all_training_data_y))

            results = self.convert_results(self.model.predict(self.data.x_test))

            report = classification_report(self.data.y_test, results, output_dict=True)

            zero = report["0"]["f1-score"]
            one = report["1"]["f1-score"]
            two = report["2"]["f1-score"]
            three = report["3"]["f1-score"]

            print(f"{len(selected_data_x)},{zero},{one},{two},{three}")
            out_file.write(f"{len(selected_data_x)},{zero},{one},{two},{three}\n")

            average = zero + one + two + three
            average /= 4
            print(average)


cont = Controller()
cont.train()
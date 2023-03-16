from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def max_diff(list):
    length = list.shape[0]
    dimension = list.shape[1]
    dist = np.zeros((length, dimension))
    for i in range(dimension):
        for j in range(length):
            for k in range(length):
                dist[j][i] += abs(list[j][i] - list[k][i])
    argmax = dist.argmax(axis=0)
    return argmax


class LDA():
    def __init__(self, train_sample, train_label,test_sample,test_label):
        self.sample = train_sample
        self.label = train_label
        self.test_sample = test_sample
        self.test_sample_num = self.test_sample.shape[0]
        self.test_sample_dim = self.test_sample.shape[1]
        self.test_label = test_label
        self.sample_num = self.sample.shape[0]  # number of samples
        self.sample_dim = self.sample.shape[1]  # number of features
        self.classes = np.unique(self.label)  # labels
        self.n_classes = len(self.classes)  # number of classes
        self.nSamples = []  # number of samples in each class
        for i in range(self.n_classes):
            temp = self.label.tolist()
            self.nSamples.append(temp.count(self.classes[i]))
        self.mean = self.get_mean()  # mean of each class
        self.Sw = self.get_Sw()  # within class scatter matrix
        self.Sb = self.get_Sb()  # between class scatter matrix
        self.eig_vals, self.eig_vecs = self.get_eig()  # eigenvalues and eigenvectors
        self.w = self.get_w()  # projection matrix
        self.w0 = self.get_w0()  # w0
        self.decision = self.make_decision()  # decision boundary

        self.projected = self.get_projection()  # projected data
        self.sample_mask = self.sample_mask(i)  # sample mask
        self.transformed = self.transformed()  # transformed data
        self.train_accuracy = self.get_train_accuracy()  # train_accuracy
        self.plot()  # plot
        self.test_sample = self.test_sample() # test_sample
        self.test_accuracy = self.get_test_accuracy()  # test_accuracy

    def get_mean(self):
        mean = np.zeros((self.n_classes, self.sample_dim))
        for i in range(self.n_classes):
            mean[i, :] = np.mean(self.sample_mask(i), axis=0)
        return mean

    def get_Sw(self):
        Sw = np.zeros((self.sample_dim, self.sample_dim))
        for i in range(self.n_classes):
            temp = self.sample_mask(i) - self.mean[i, :]
            Sw += np.dot(temp.T, temp)
        return Sw

    def get_Sb(self):
        Sb = np.zeros((self.sample_dim, self.sample_dim))
        for i in range(self.n_classes):
            temp = self.mean[i, :] - np.mean(self.sample, axis=0)
            Sb += self.nSamples[i] * np.dot(temp.T, temp)
        return Sb

    def get_eig(self):
        eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(self.Sw), self.Sb))
        return eig_vals, eig_vecs

    def get_w(self):
        eig_pairs = [(np.abs(self.eig_vals[i]), self.eig_vecs[:, i]) for i in range(len(self.eig_vals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        w = np.hstack((eig_pairs[i][1].reshape(self.sample_dim, 1) for i in range(len(self.eig_vals))))
        return w

    def get_projection(self):
        projection = np.zeros((self.n_classes, self.n_classes - 1))
        for i in range(self.n_classes):
            projection[i, :] = np.dot(self.mean[i, :], self.w[:, 0:self.n_classes - 1])
        return projection

    def sample_mask(self, i):
        temp = np.hstack((self.label == self.classes[i]) for j in range(self.sample_dim))
        masked_sample = self.sample[temp].reshape(self.nSamples[i], self.sample_dim)
        return masked_sample

    def transformed(self):
        transformed_on_w1 = np.zeros((self.sample_num, self.n_classes - 1))
        transformed_on_w2 = np.zeros((self.sample_num, self.n_classes - 1))
        transformed = np.zeros((self.sample_num, self.n_classes - 1))
        for i in range(self.sample_num):
            transformed_on_w1[i, :] = np.dot(self.sample[i, :], self.w[:, 0])
            transformed_on_w2[i, :] = np.dot(self.sample[i, :], self.w[:, 1])
            transformed[i, :] = np.dot(self.sample[i, :], self.w[:, 0:self.n_classes - 1])
        return transformed_on_w1, transformed_on_w2, transformed

    def get_w0(self):
        # w0 = np.zeros((self.n_classes-1, 1))
        w0 = np.array([1.25, 0.6])
        return w0

    def make_decision(self):
        decision = np.zeros((self.sample_num, 1))
        for i in range(self.sample_num):
            for j in range(2):
                if (np.dot(self.sample[i, :], self.w[:, 1]) + self.w0[0] > 0) & (
                        np.dot(self.sample[i, :], self.w[:, 1]) + self.w0[1] < 0):
                    decision[i] = 3
                elif (np.dot(self.sample[i, :], self.w[:, 1]) + self.w0[0] < 0) & (
                        np.dot(self.sample[i, :], self.w[:, 1]) + self.w0[1] > 0):
                    decision[i] = 2
                elif (np.dot(self.sample[i, :], self.w[:, 1]) + self.w0[0] < 0) & (
                        np.dot(self.sample[i, :], self.w[:, 1]) + self.w0[1] < 0):
                    decision[i] = 1
                elif (np.dot(self.sample[i, :], self.w[:, 1]) + self.w0[0] > 0) & (
                        np.dot(self.sample[i, :], self.w[:, 1]) + self.w0[1] > 0):
                    if (np.dot(self.sample[i, :], self.w[:, 0]) + self.w0[0]) > (
                            np.dot(self.sample[i, :], self.w[:, 1]) + self.w0[1]):
                        decision[i] = 3
                    elif (np.dot(self.sample[i, :], self.w[:, 0]) + self.w0[0]) < (
                            np.dot(self.sample[i, :], self.w[:, 1]) + self.w0[1]):
                        decision[i] = 2

        return decision

    def plot(self):
        colors = ['r', 'g', 'b']
        plt.figure(1, figsize=(10, 5))
        plt.title('Projection on w1')
        for points, color in zip(self.transformed[0], self.label - 1):
            plt.scatter(points[0], 0, color=np.array(colors)[color], alpha=0.5)
        plt.legend(["Class 1", "Class 2", "Class 3"])
        plt.show()
        plt.figure(2, figsize=(10, 5))
        plt.title('Projection on w2')
        for points, color in zip(self.transformed[1], self.label - 1):
            plt.scatter(points[0], 0, color=np.array(colors)[color], alpha=0.5)
        plt.legend(["Class 1", "Class 2", "Class 3"])
        plt.show()
        plt.figure(3, figsize=(7, 7))
        plt.title('Projection on w1 and w2')
        for points, color in zip(self.transformed[2], self.label - 1):
            plt.scatter(points[0], points[1], color=np.array(colors)[color], alpha=0.5)
        plt.legend(["Class 1", "Class 2", "Class 3"])
        plt.show()

    def get_train_accuracy(self):
        correct = 0
        for i in range(self.sample_num):
            if self.decision[i] == self.label[i]:
                correct += 1
        accuracy = (correct * 100) / self.sample_num
        float = '{:.2f}'.format(accuracy)
        return float

    def get_test_accuracy(self):
        correct = 0
        for i in range(self.test_sample_num):
            if self.decision[i] == self.test_label[i]:
                correct += 1
        accuracy = (correct * 100) / self.test_sample_num
        float = '{:.2f}'.format(accuracy)
        return float

if __name__ == '__main__':
    data = loadmat('Data_Train.mat')
    labels = loadmat('Label_Train.mat')

    data = np.array(data['Data_Train'])
    labels = np.array(labels['Label_Train'])

    lda = LDA(data, labels)
    projected = LDA.get_projection(lda)
    # print(projected)
    transformed = LDA.transformed(lda)

    # ad = max_diff(np.array([[-0.8, -2.3], [-2.3, -0.4], [-1.7, -0.8]]))
    # print(ad)
    print(LDA.get_accuracy(lda), '%')

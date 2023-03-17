import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


class MAP():
    def __init__(self, train_sample, train_label, test_sample=np.array([[]]), test_label=np.array([[]])):
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
        self.Cov = self.get_Cov()  # Covariance matrix
        # self.decision = self.make_decision()  # decision boundary
        self.sample_mask = self.sample_mask(i)  # sample mask
        self.Px_omega = self.get_Px_omega()  # Px_omega
        self.P_omega = self.get_P_omega() # P_omega
        self.gx = self.get_gx() # gx
    def get_mean(self):
        mean = np.zeros((self.n_classes, self.sample_dim))
        for i in range(self.n_classes):
            mean[i, :] = np.mean(self.sample_mask(i), axis=0)
        return mean

    def sample_mask(self, i):
        temp = np.hstack((self.label == self.classes[i]) for j in range(self.sample_dim))
        masked_sample = self.sample[temp].reshape(self.nSamples[i], self.sample_dim)
        return masked_sample

    def get_Cov(self):
        Cov = np.zeros((self.n_classes, self.sample_dim, self.sample_dim))
        for i in range(self.n_classes):
            temp = self.sample_mask(i) - self.mean[i, :]
            Cov[i] += np.dot(temp.T, temp)
        return Cov

    def get_Px_omega(self):
        px = np.zeros((self.n_classes, self.sample_dim))
        for i in range(self.n_classes):
            px[i, :] = np.mean(self.sample_mask(i), axis=0)
        return px
    # todo: px_omega, p_omega and gx are not determined yet

if __name__ == '__main__':
    data = loadmat('Data_Train.mat')
    labels = loadmat('Label_Train.mat')

    data = np.array(data['Data_Train'])
    labels = np.array(labels['Label_Train'])

    MAP(data, labels)

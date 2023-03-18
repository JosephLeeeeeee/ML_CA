import numpy as np
from scipy.io import loadmat


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
        self.nSamples = [self.label.tolist().count(cls) for cls in self.classes]  # number of samples in each class
        self.mean = self.get_mean()  # mean of each class
        self.Cov = self.get_Cov()  # Covariance matrix

        self.sample_mask = self.sample_mask(i=0)  # sample mask
        self.Px_omega = self.get_Px_omega()  # Px_omega
        self.P_omega = self.get_P_omega()  # P_omega
        self.gx = self.get_gx()  # gx
        self.decision = self.make_decision()  # decision boundary
        self.train_accuracy = self.get_train_accuracy()  # train_accuracy

    def get_mean(self):
        mean = np.array([np.mean(self.sample_mask(i), axis=0) for i in range(self.n_classes)])
        return mean

    def sample_mask(self, i):
        temp = np.hstack((self.label == self.classes[i]) for j in range(self.sample_dim))
        masked_sample = self.sample[temp].reshape(self.nSamples[i], self.sample_dim)
        return masked_sample

    def get_Cov(self):
        Cov = np.zeros((self.n_classes, self.sample_dim, self.sample_dim))
        for i in range(self.n_classes):
            temp = self.sample_mask(i) - self.mean[i, :]
            Cov[i] += 1 / self.nSamples[i] * np.dot(temp.T, temp)
        return Cov

    def get_Px_omega(self):
        Px_omega = np.zeros((self.sample_num, self.n_classes))
        for i in range(self.n_classes):
            aaa = 1 / ((2 * np.pi) ** (self.sample_dim / 2) * np.linalg.det(self.Cov[i]) ** (1 / 2))
            bbb = np.dot((self.sample - self.mean[i, :]), np.linalg.inv(self.Cov[i]))
            ccc = np.zeros((self.sample_num, 1))
            for row in enumerate(bbb):
                ccc[row[0]] = (np.dot(row[1], (self.sample[row[0], :] - self.mean[i, :])))
            ddd = np.exp(-1 / 2 * ccc)
            Px_omega[:, i] = aaa * ddd[:, 0]
            # Px_omega[:,i] = 1/((2*np.pi)**(self.sample_dim/2)*np.linalg.det(self.Cov[i])**(1/2))*np.exp(-1/2*np.dot(np.dot((self.sample-self.mean[i,:]).T, np.linalg.inv(self.Cov[i])), (self.sample-self.mean[i,:])))
        return Px_omega

    def get_P_omega(self):
        P_omega = np.zeros((self.n_classes, 1))
        for i in range(self.n_classes):
            P_omega[i] = self.nSamples[i] / self.sample_num
        return P_omega

    def get_gx(self):
        gx = np.zeros((self.sample_num, self.n_classes))
        for i in range(self.n_classes):
            gx[:, i] = self.Px_omega[:, i] * self.P_omega[i]
        return gx

    def make_decision(self):
        decision = (np.argmax(self.gx, axis=1) + 1).reshape(self.sample_num, 1)
        return decision

    def get_train_accuracy(self):
        correct = sum(np.all(self.decision == self.label, axis=1))
        accuracy = (correct * 100) / self.sample_num
        float = '{:.2f}'.format(accuracy)
        return float

if __name__ == '__main__':
    data = loadmat('Data/Data_Train.mat')
    labels = loadmat('Data/Label_Train.mat')

    data = np.array(data['Data_Train'])
    labels = np.array(labels['Label_Train'])

    map = MAP(data, labels)
    print('training accuracy is: ', MAP.get_train_accuracy(map), '%')
from scipy.io import loadmat
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
    def __init__(self, train_sample, train_label, test_sample=np.array([[]]), test_label=np.array([[]])):
        # training parameters
        self.sample = train_sample
        self.label = train_label
        self.sample_num = self.sample.shape[0]  # number of samples
        self.sample_dim = self.sample.shape[1]  # number of features
        self.classes = np.unique(self.label)  # labels
        self.n_classes = len(self.classes)  # number of classes
        self.nSamples = [self.label.tolist().count(cls) for cls in self.classes]  # number of samples in each class

        # testing parameters
        self.test_sample = test_sample
        self.test_label = test_label
        self.test_sample_num = self.test_sample.shape[0]
        self.test_sample_dim = self.test_sample.shape[1]

        # functions
        self.mean = self.get_mean()  # mean of each class
        self.Sw = self.get_Sw()  # within class scatter matrix
        self.Sb = self.get_Sb()  # between class scatter matrix
        self.eig_vals, self.eig_vecs = self.get_eig()  # eigenvalues and eigenvectors
        self.w = self.get_w()  # projection matrix
        self.transformed = self.transformed()  # transformed data
        self.projected = self.get_projection()  # projected data
        self.decision = self.make_decision()  # decision boundary
        self.sample_mask = self.sample_mask(i=0)  # sample mask
        self.train_accuracy = self.get_train_accuracy()  # train_accuracy
        self.plot()  # plot

        self.test_accuracy = self.get_test_accuracy()  # test_accuracy

    def get_mean(self):
        mean = np.array([np.mean(self.sample_mask(i), axis=0) for i in range(self.n_classes)])
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
            tempp = temp.reshape(1, self.sample_dim)
            Sb += self.nSamples[i] * np.dot(tempp.T, tempp)
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
        projection = np.array([np.dot(self.mean[i, :], self.w[:, 0:self.n_classes - 1]) for i in range(self.n_classes)])
        return projection

    def sample_mask(self, i):
        temp = np.hstack((self.label == self.classes[i]) for j in range(self.sample_dim))
        masked_sample = self.sample[temp].reshape(self.nSamples[i], self.sample_dim)
        return masked_sample

    def transformed(self):
        transformed = np.zeros((self.sample_num, self.n_classes - 1))
        for i in range(self.sample_num):
            transformed[i, :] = np.dot(self.sample[i, :], self.w[:, 0:self.n_classes - 1])
        return transformed

    def make_decision(self):
        # projected to 2D,w1 and w2
        # distance = np.zeros((self.sample_num, self.n_classes))
        # for i in range(self.sample_num):
        #     for j in range(self.n_classes):
        #         distance[i, j] = np.sqrt(sum((self.transformed[i, :] - self.projected[j, :]) ** 2))
        # decision = np.argmin(distance, axis=1) + 1
        # decision = decision.reshape(self.sample_num, 1)

        # projected to 1D,w1 (which has a better result,LMAO)
        distance = np.zeros((120,3))
        for i in range(120):
            for j in range(3):
                distance[i,j] += (self.transformed[i,0] - self.projected[j, 0]) ** 2
        decision = np.argmin(distance, axis=1) + 1
        decision = decision.reshape(120, 1)
        return decision

    def plot(self):
        sample_restack = []

        for i in range(self.n_classes):
            temp = np.hstack((self.label == self.classes[i]))
            masked_sample = self.transformed[temp].reshape(self.nSamples[i], 2)
            sample_restack.append(masked_sample)
        sample_restack = np.concatenate(sample_restack, axis=0)

        indices = np.concatenate([np.repeat(i, self.nSamples[i]) for i in range(self.n_classes)])
        label_restack = indices.reshape(self.sample_num, 1)

        step = np.arange(-self.sample_num / 2, self.sample_num / 2, 1).reshape(self.sample_num, 1)

        colors = ['r', 'g', 'b']
        color_labels = ["Class 1", "Class 2",  "Class 3"]
        added_label = []

        plt.figure(1, figsize=(10, 5))
        plt.title('Projection on w1')
        for steps,points, color in zip(step,sample_restack[:,0], label_restack):
            if color not in added_label:
                plt.scatter(steps, points, color=np.array(colors)[color], alpha=0.5, label=color_labels[color[0]])
                added_label.append(color)
            else:
                plt.scatter(steps, points, color=np.array(colors)[color], alpha=0.5)
        plt.legend()
        plt.ylabel('w1')
        plt.show()

        added_label = []
        plt.figure(2, figsize=(10, 5))
        plt.title('Projection on w2')
        for steps,points, color in zip(step,sample_restack[:,1], label_restack):
            if color not in added_label:
                plt.scatter(steps, points, color=np.array(colors)[color], alpha=0.5, label=color_labels[color[0]])
                added_label.append(color)
            else:
                plt.scatter(steps, points, color=np.array(colors)[color], alpha=0.5)
        plt.legend()
        plt.ylabel('w2')
        plt.show()

        added_label = []
        plt.figure(3, figsize=(7, 7))
        plt.title('Projection on w1 and w2')
        for points, color in zip(sample_restack, label_restack):
            if color not in added_label:
                plt.scatter(points[0], points[1], color=np.array(colors)[color], alpha=0.5 ,label=color_labels[color[0]])
                added_label.append(color)
            else:
                plt.scatter(points[0], points[1], color=np.array(colors)[color], alpha=0.5 )
        plt.legend()
        plt.xlabel('w1')
        plt.ylabel('w2')
        plt.show()

    def get_train_accuracy(self):
        correct = sum(np.all(self.decision == self.label, axis=1))
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
    data = loadmat('Data/Data_Train.mat')
    labels = loadmat('Data/Label_Train.mat')

    data = np.array(data['Data_Train'])
    labels = np.array(labels['Label_Train'])

    lda = LDA(data, labels)

    print('training accuracy is:', LDA.get_train_accuracy(lda), '%')

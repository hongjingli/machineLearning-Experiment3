import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import math
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.h_list = []
        self.alpha_list = []
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).
        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        num=len(y)
        #初始化分布w
        w = np.empty(shape=num, dtype=float)
        for i in range(num):
            w[i] = 1.0 /num
        for t in range(self.n_weakers_limit):
            #如果使用固有的weak_classifier，会累计训练的过程，导致结果较差
            #故使用新的决策树进行(X,y)在分布w上的训练
            #clf = self.weak_classifier
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(X, y,w)

            #计算epsilon和统计正确个数（用以观察训练过程）
            epsilon = 0
            y_test=clf.predict(X)
            print(y_test)
            print(y)
            count=0
            for i in range(len(y)):
                if y_test[i]!= y[i]:
                    epsilon =epsilon+ w[i]
                else:
                    count=count+1
            print(count)

            if epsilon > 0.5: break
            #计算alpha值
            self.alpha_list.append(0.5 * math.log((1-epsilon)/epsilon))
            z_sum = 0
            for i in range(len(y)):
                w[i] = w[i] * math.exp(-self.alpha_list[t]*y_test[i]*y[i])
                z_sum +=w[i]
            #除以z_sum进行归一化
            for i in range(len(y)):
                w[i] /= z_sum
            self.h_list.append(clf)
        pass


    def predict_scores(self, X):

        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        res =[]
        for i in range(X.shape[0]):
            s = 0
            for j in range(len(self.h_list)):
                s += self.h_list[j].predict(X[i].reshape(1,-1)) * self.alpha_list[j]
            res.append(s)
        return res

        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        res = np.array(self.predict_scores(X))
        res[res>=threshold]=1
        res[res<threshold]=-1
        return res

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

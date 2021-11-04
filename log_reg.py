import numpy as np
import pandas as pd 
import wandb

wandb.login()
wandb.init(project="ift6390-extreme-weather-events")


class MultiClass_LogisticRegression():
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-5, verbose=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.coef_ = None
        self.intercept_ = None
        self.loss_ = None
        self.n_iter_ = None

    def fit(self, X, y):

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Initialize parameters
        self.coef_ = np.zeros((n_classes, n_features))
        self.intercept_ = np.zeros(n_classes)

        # Initialize loss
        self.loss_ = []

        for it in range(self.max_iter):
            # Compute softmax scores
            scores = np.dot(X, self.coef_.T)
            scores += self.intercept_
            scores -= np.max(scores, axis=1, keepdims=True)
            scores_exp = np.exp(scores)
            scores_exp_sum = np.sum(scores_exp, axis=1, keepdims=True)
            softmax_scores = scores_exp / scores_exp_sum

            # Compute loss
            loss = -np.sum(np.log(softmax_scores[range(n_samples), y.astype(int)]))
            self.loss_.append(loss)

            # Compute gradient
            dscores = softmax_scores
            dscores[range(n_samples), y.astype(int)] -= 1
            dscores /= n_samples

            # Update parameters
            self.coef_ -= self.learning_rate * np.dot(dscores.T, X)
            self.intercept_ -= self.learning_rate * np.sum(dscores, axis=0)

            wandb.log({"loss": loss})
            wandb.log({'learning_rate': self.learning_rate})
            wandb.log({'Iterations': it})
            wandb.log({'Co-efficient': self.coef_})
            wandb.log({'Intercept': self.intercept_})

            # Stop iterating if loss is small
            if loss < self.tol:
                break

        self.n_iter_ = it + 1

        return self

    def predict(self, X):
        scores = np.dot(X, self.coef_.T)
        scores += self.intercept_
        return np.argmax(scores, axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    data = pd.read_csv('./data/train.csv')
    data = data.drop(['S.No'], axis=1)
    # data = data.droxp_duplicates()
    data = data.values

    np.random.seed(2)
    inds = np.arange(data.shape[0])
    np.random.shuffle(inds)
    num_train = int(0.8 * data.shape[0])
    train_inds = inds[:num_train]
    val_inds = inds[num_train:]

    trainset = data[train_inds] 
    valset = data[val_inds]

    # Normalize train set to mean 0 and standard deviation 1 feature-wise.
    # Apply the same transformation to the test set.
    mu = trainset[:, :-1].mean(axis=0)
    sigma = trainset[:, :-1].std(axis=0)
    trainset[:, :-1] = (trainset[:, :-1] - mu)/sigma
    valset[:, :-1] = (valset[:, :-1] - mu)/sigma

    train_X = trainset[:, :-1]
    train_y = trainset[:, -1]

    val_X = valset[:, :-1]
    val_y = valset[:, -1]

    # create a logistic regression classifier
    clf = MultiClass_LogisticRegression(learning_rate=0.01, max_iter=1000, tol=1e-5, verbose=False)
    clf.fit(train_X, train_y)

    print("Training set accuracy: {:.2f}".format(clf.score(train_X, train_y)))
    print("Validation set accuracy: {:.2f}".format(clf.score(val_X, val_y)))

    test_df = pd.read_csv('./data/test.csv')
    test_data = test_df.drop(['S.No'], axis=1)
    test_data = test_data.values

    test_data = (test_data - mu)/sigma

    # make predictions on test set
    y_pred = clf.predict(test_data)
    print(y_pred)

    # save results
    # submission = pd.DataFrame({'S.No': test_df.index,'LABELS':y_pred})
    # filename = 'Submission_Diganta_20213809_logreg.csv'

    # submission.to_csv(filename,index=False)

    # print('Saved file: ' + filename)


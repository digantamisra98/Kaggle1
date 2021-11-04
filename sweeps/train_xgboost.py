import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import wandb

class XGBoost_classifier(object):

    def __init__(self, train_path = "", test_path = ""):
        self.train_path = train_path
        self.test_path = test_path
        self.x_train, self.x_test, self.y_train, self.y_test = self.load_data()
        self.xgboost_classifier()
        self.test()

    def load_data(self):
        train_df = pd.read_csv(self.train_path, index_col="S.No")
        test_df = pd.read_csv(self.test_path, index_col="S.No")
        X = train_df.iloc[:, :-1]
        Y = train_df.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test

    def xgboost_classifier(self):
        params = {
                    'objective':'multi:softmax',
                    'max_depth': wandb.config.max_depth,
                    'alpha': wandb.config.alpha,
                    'learning_rate': wandb.config.learning_rate,
                    'n_estimators':wandb.config.n_estimators,
                    'scale_pos_weight':1}    

        self.xgb_clf = XGBClassifier(**params)
        self.xgb_clf.fit(self.x_train, self.y_train)

    def test(self):
        y_pred = self.xgb_clf.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy: ", accuracy)
        wandb.log({'Accuracy': accuracy})
        return accuracy



if __name__ == "__main__":
    wandb.init(project="xgboost_classifier")
    XGBoost_classifier(train_path = "../data/train.csv", test_path = "../data/test.csv")



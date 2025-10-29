# Practice implementations of supervised learning algorithms
# TODO: Implement the algorithms below

from .logistic_regression import LogisticRegression
from .regression import LinearRegression, PolynomialRegression, LassoRegression, RidgeRegression, ElasticNet
from .k_nearest_neighbors import KNN
from .decision_tree import DecisionTree, RegressionTree, ClassificationTree
from .naive_bayes import NaiveBayes
from .perceptron import Perceptron
from .random_forest import RandomForest
from .support_vector_machine import SupportVectorMachine
import sys
import numpy as np

digitfnames={
        'trf':'../hw12data/digitsDataset/trainFeatures.csv',
        'trl':'../hw12data/digitsDataset/trainLabels.csv',
        'tef':'../hw12data/digitsDataset/testFeatures.csv',
        'vl':'../hw12data/digitsDataset/valLabels.csv',
        'vf':'../hw12data/digitsDataset/valFeatures.csv',
        }

emailfnames={
        'trf':'../hw12data/emailDataset/trainFeatures.csv',
        'trl':'../hw12data/emailDataset/trainLabels.csv',
        'tef':'../hw12data/emailDataset/testFeatures.csv',
        'vl':'../hw12data/emailDataset/valLabels.csv',
        'vf':'../hw12data/emailDataset/valFeatures.csv',
        }

def load_digit_training_data():
    features = np.loadtxt(digitfnames['trf'],delimiter=',')
    labels = np.loadtxt(digitfnames['trl'],delimiter=',')
    return features,labels
def load_digit_validation_data():
    features = np.loadtxt(digitfnames['vf'],delimiter=',')
    labels = np.loadtxt(digitfnames['vl'],delimiter=',')
    return features,labels
def load_digit_test_data():
    features = np.loadtxt(digitfnames['tef'],delimiter=',')
    return features

def load_email_training_data():
    features = np.loadtxt(emailfnames['trf'],delimiter=',')
    labels = np.loadtxt(emailfnames['trl'],delimiter=',')
    return features,labels
def load_email_validation_data():
    features = np.loadtxt(emailfnames['vf'],delimiter=',')
    labels = np.loadtxt(emailfnames['vl'],delimiter=',')
    return features,labels
def load_email_test_data():
    features = np.loadtxt(emailfnames['tef'],delimiter=',')
    return features

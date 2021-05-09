import keras
import tensorflow as tf
import numpy as np
seed = 1
rand = np.random.RandomState(seed)
np.random.seed(1)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import to_categorical

# Setting the graph-level random seed.
tf.random.set_seed(1)
def generateSyntheticDataset(numberOfSample):
        stdev = rand.normal(0,1,numberOfSample)
        x0 = rand.uniform(0,1,numberOfSample)
        #print(x0)
        x1 = rand.uniform(0,1,numberOfSample)
        x2 = rand.uniform(0,1,numberOfSample)
        x3 = rand.uniform(0,1,numberOfSample)
        x4 = rand.uniform(0,1,numberOfSample)
        ###irrelevant
        x5 = rand.uniform(0,1,numberOfSample)
        x6 = rand.uniform(0,1,numberOfSample)
        x7 = rand.uniform(0,1,numberOfSample)
        x8 = rand.uniform(0,1,numberOfSample)
        x9 = rand.uniform(0,1,numberOfSample)
        ##higly correleted
        x10 = x0 + rand.normal(0,0.01,numberOfSample)
        x11 = x1 + rand.normal(0,0.01,numberOfSample)
        x12 = x2 + rand.normal(0,0.01,numberOfSample)
        x13 = x3 + rand.normal(0,0.01,numberOfSample)
        y = 10*np.sin(np.pi*x0*x1) - 20*(x2 -0.5)**2 + 10*x3 + 5*x4 + stdev
        data = []
        for i in x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13:
            data.append(i)
        data = np.array(data)
        mean = np.mean(y)
        new_y = []
        for i in y:
            if i > mean:
                new_y.append(0)
            else:
                new_y.append(1)
        y = np.array(new_y)
        yb = y
        y =  to_categorical(y)
        df = pd.DataFrame(data=[], columns=[i for i in range(14)])
        return (data.T,y, df, yb)


def generateCovidDataset():
    #https://github.com/stccenter/COVID-19/tree/master/prediction/patiant-level%20fatality/data
    dataset = pd.read_csv("small records.csv")
    y = pd.DataFrame(data=dataset["death_binary"], columns=["death_binary"])
    dataset.drop("death_binary",axis=1,inplace=True)
    dataset.drop("DateOfOnsetSymptoms",axis=1,inplace=True)
    dataset.drop("ID",axis=1,inplace=True)
    dataset.drop("NA",axis=1,inplace=True)
    dataset.drop("Longitiude",axis=1,inplace=True)
    dataset.drop("Latitude",axis=1,inplace=True)

    print(dataset.columns)
    dataset['age'] = dataset['age'].replace(["0-6"],["3"])
    dataset['age'] = dataset['age'].replace(["0-10"],["5"])
    dataset['age'] = dataset['age'].replace(["18-60"],["35"])
    dataset['age'] = dataset['age'].replace(["30-35"],["33"])
    dataset['age'] = dataset['age'].replace(["13-19"],["15"])
    dataset['age'] = dataset['age'].replace(["20-29"],["25"])
    dataset['age'] = dataset['age'].replace(["30-39"],["25"])
    dataset['age'] = dataset['age'].replace(["40-49"],["25"])
    dataset['age'] = dataset['age'].replace(["50-59"],["25"])
    dataset['age'] = dataset['age'].replace(["50-69"],["30"])
    dataset['age'] = dataset['age'].replace(["60-69"],["25"])
    dataset['age'] = dataset['age'].replace(["70-79"],["25"])
    dataset['age'] = dataset['age'].replace(["80-89"],["25"])
    dataset['age'] = dataset['age'].replace(["90-99"],["25"])
    dataset['age'] = dataset['age'].replace(["100-109"],["25"])
    dataset.drop("age",axis=1,inplace=True)

    y =  to_categorical(y)
    df = dataset
    dataset = np.asarray(dataset).astype('float32')
    return dataset, y, df

def generateCardioDataset():
    dataset = pd.read_csv("cardio_train.csv", delimiter=';')
    y = pd.DataFrame(data=dataset["cardio"], columns=["cardio"])
    dataset.drop("cardio",axis=1,inplace=True)
    dataset.drop("id",axis=1,inplace=True)
    df = dataset
    dataset = np.asarray(dataset).astype('float32')
    y =  to_categorical(y)
    return dataset, y, df

def generateDivorceDataset():
    #https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set
    dataset = pd.read_csv("divorce.csv", delimiter=';')
    y = pd.DataFrame(data=dataset["Class"], columns=["Class"])
    dataset.drop("Class",axis=1,inplace=True)
    df = dataset
    dataset = np.asarray(dataset).astype('float32')
    y =  to_categorical(y)
    return dataset, y, df

    
def generateChildhoodTumor():
    #http://csse.szu.edu.cn/staff/zhuzx/Datasets.html
    raw_data = loadarff('SRBCT.arff')
    df= pd.DataFrame(raw_data[0])
    y = pd.DataFrame(data=df["CLASS"], columns=["CLASS"])
    df.drop("CLASS",axis=1,inplace=True)
    dataset = np.asarray(df.values).astype('float32')
    y = y.values
    y =  to_categorical(y)
    return dataset, y, df

def generateGDataset(ndata, nlab, col=""):
    dataset = pd.read_csv(ndata, index_col=0)
    y = pd.read_csv(nlab, index_col=0)
    y = y["x"].replace(-1, 0)
    print(y)
    df = dataset
    if col != "":
        df.columns = col
    print(df)
    dataset = np.asarray(dataset).astype('float32')
    y = np.asarray(y).astype('float32')

    y =  to_categorical(y)
    return dataset, y, df

breast_feature = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
 'mean smoothness', 'mean compactness', 'mean concavity',
 'mean concave points', 'mean symmetry', 'mean fractal dimension',
 'radius error', 'texture error', 'perimeter error', 'area error',
 'smoothness error', 'compactness error', 'concavity error',
 'concave points error' ,'symmetry error', 'fractal dimension error',
 'worst radius', 'worst texture', 'worst perimeter', 'worst area',
 'worst smoothness' ,'worst compactness', 'worst concavity',
 'worst concave points', 'worst symmetry' ,'worst fractal dimension']
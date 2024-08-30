import numpy as np
import os
# Get the script directory
script_directory = os.path.dirname(os.path.realpath(__file__))

# Change the current working directory to the script directory
os.chdir(script_directory)

# Load data from files
X_seen=np.load('X_seen.npy',encoding='bytes',allow_pickle=True)
Xtest=np.load('Xtest.npy',encoding='bytes',allow_pickle=True)
Ytest=np.load('Ytest.npy',encoding='bytes',allow_pickle=True)
class_attributes_seen=np.load('class_attributes_seen.npy',encoding='bytes',allow_pickle=True)
class_attributes_unseen=np.load('class_attributes_unseen.npy',encoding='bytes',allow_pickle=True)
meanOfClasses=np.zeros((40,4096))
countOfclassElements=np.zeros(40)
l=[0.01,0.1,1,10,20,50,100]

# Calculate the mean of seen classes
for i in range(40):
    meanOfClasses[i] = np.sum(X_seen[i], axis=0)
    countOfclassElements[i] = X_seen[i].shape[0]
meanOfClasses /= countOfclassElements.reshape(-1, 1)


def findW(A,lamda,M):        #A=class_attributes_seen            M=meanofclasses
    I=np.eye(A.shape[1])
    X=np.dot(A.T, A)
    lambda_I = lamda*I
    Y = np.linalg.inv(X+lambda_I)
    Z = np.dot(A.T, M)
    W = np.dot(Y, Z)
    return W


def findMean(W,a):   #w=85x4096  a=85x1
    mean = np.dot(W.T, a)
    return mean


accuracy=[]

# Iterate through different lambda values
for lamda in l:
    w=findW(class_attributes_seen,lamda,meanOfClasses)
    
    mean=[]
    for i in range(len(class_attributes_unseen)):
        mean.append(findMean(w,class_attributes_unseen[i]))
    
    mean=np.array(mean)
    correctPrediction=0
    for i in range(len(Xtest)):
        ans=float('inf')
        ExpClass=-1
        for j in range(len(mean)):
            y=np.linalg.norm(Xtest[i]-mean[j])
            if(y<ans):
                ans=y
                ExpClass=j+1
        if(ExpClass==Ytest[i]):
            correctPrediction+=1
    
    accuracy.append(correctPrediction/len(Xtest))
    print("Accuracy for lamda=",lamda," is ",accuracy[-1]*100,"%")




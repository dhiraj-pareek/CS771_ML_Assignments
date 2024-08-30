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
similarityVector=np.zeros((10,40))
meanOfUnseenClasses=np.zeros((10,4096), dtype=float)

# Calculate the mean of seen classes
for i in range(40):
    meanOfClasses[i] = np.sum(X_seen[i], axis=0)
    countOfclassElements[i] = X_seen[i].shape[0]
meanOfClasses /= countOfclassElements.reshape(-1, 1)

# Calculate similarity between seen and unseen class attributes
for i in range(10):
    for j in range(40):
            similarityVector[i,j]=np.sum(class_attributes_unseen[i]*class_attributes_seen[j])
similarityVector /= similarityVector.sum(axis=1, keepdims=True)

# Calculate mean of unseen classes
for i in range(10):
    for j in range(40):
        meanOfUnseenClasses[i] += similarityVector[i, j] * meanOfClasses[j]

# Classify test data
correctPrediction=0
for i in range(len(Xtest)):
    ans=float('inf')
    ExpClass=-1
    for j in range(len(meanOfUnseenClasses)):
          y=np.linalg.norm(Xtest[i]-meanOfUnseenClasses[j])
          if(y<ans):
               ans=y
               ExpClass=j+1
    if(ExpClass==Ytest[i]):
         correctPrediction+=1
         
# Calculate accuracy
accuracy=(correctPrediction/len(Xtest))*100


print("Accuracy is :",accuracy,"%")
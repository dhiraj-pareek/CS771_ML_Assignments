import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

data=open("ridgetrain.txt","r")
d=data.read().split()
# print(d)
dataa=open("ridgetest.txt","r")
data1=[]
for i in range(0,len(d),2):
    data1.append([float(d[i]),float(d[i+1])])

def kernel(x,y):
    return math.exp(-0.1*(np.linalg.norm(x-y))**2)

k=[]

for i in range(len(data1)):
    l=[]
    for j in range(len(data1)):
        l.append(kernel(data1[i][0],data1[j][0]))
    k.append(l)


#w=(k+lamdaIn)-1 (Y)
lamda=1
for i in range(len(data1)):
    k[i][i]+=lamda
k=np.linalg.inv(k)


w=[]
for i in range(len(k)):
    l=0
    for j in range(len(k[i])):
        l+=k[i][j]*data1[j][1]
    w.append(l)

d2=dataa.read().split()
testData=[]
for i in range(0,len(d2),2):
    testData.append([float(d2[i]),float(d2[i+1])])
w=np.array(w)
pred=[]
for i in range(len(testData)):
    kx=[]
    for j in range(len(data1)):
        kx.append(kernel(testData[i][0],data1[j][0]))
    pred.append(np.dot(w.T,kx))
x=[]
y=[]
for i in range(len(testData)):
    x.append(testData[i][0])
    y.append(testData[i][1])

plt.scatter(x, y, marker='o', color='b', label='Scatter Plot')
plt.scatter(x, pred, marker='o', color='g', label='Scatter Plot')

# Add labels and a title
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title('Simple Scatterplot')

# Add a legend (if needed)
plt.legend()

# Show the plot
plt.show()


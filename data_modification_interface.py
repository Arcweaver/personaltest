import streamlit as st
import pandas as pd
import datetime
from datetime import date
import os
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from io import StringIO
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def dfToDict(df, col1, col2):
    return {a: b for a, b, include in zip(df[col1], df[col2], df['Include']) if include}




#variable initialization
input_dim  = 4 
learning_rate = 0.01
num_epochs = 1000
classFilePath = "M:/People/James/test/python/FYP_stuff/data/flower.csv"
try:
    originalTrainData = pd.read_csv('put_path_here')
except:
    originalTrainData = pd.DataFrame()

try:
    flower_mapping_frame = pd.read_csv(classFilePath)
except:
    flower_label_data = {
        'Flower': ['Setosa', 'Versicolour', 'Virginica', 'Sunflower'],
        'Id': [0, 1, 2, 3],
        'Include': [True, True, True, False]
    }
    flower_mapping_frame = pd.DataFrame(flower_label_data)
    flower_mapping_frame.to_csv(classFilePath, index=False)

label_mapping = dfToDict(flower_mapping_frame, 'Id', 'Flower')
output_dim = len(label_mapping)

def draw():
    st.header("Modify classes and test data here")
    st.divider()
    st.subheader("Classes:")
    edited_df = st.data_editor(
        flower_mapping_frame,
        column_config={
            'Id': 'Id',
            'Flower': 'Flowers',
            'Include': st.column_config.CheckboxColumn(
                'Include?',
                help="Select to include this element",
                default=False,
            ),
        },
        num_rows="dynamic"
    )

    if st.button("Confirm Change"):
        edited_df.to_csv(classFilePath, index=False)
        label_mapping = dfToDict(edited_df, 'Id', 'Flower')
        output_dim = len(label_mapping)
        st.write("Change saved")

    # upload test data and combine csv
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        newTrainData = pd.read_csv(uploaded_file)
        originalTrainData = originalTrainData.merge(newTrainData, how='outer')






#test dataset
iris_temp = datasets.load_iris()
iris = pd.DataFrame(iris_temp.data, columns=iris_temp.feature_names)
iris['target'] = iris_temp.target
X = iris.drop(["target"],axis=1).values
y = iris["target"].values
scaler = StandardScaler()



#for real data
# originalData = pd.DataFrame(originalTrainData.data, columns=originalTrainData.feature_names)
# originalData['target'] = originalTrainData.target
# X = originalData.drop(["target"],axis=1).values
# y = originalData["target"].values
# scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


def read_label_mapping():
    pass






# # ui
# def draw():
#     st.write("This is iris flower v2 using PyTorch")

#     sepalLength = st.slider(
#         "Select sepal length (in cm)",
#         0.0, 10.0)

#     sepalWidth = st.slider(
#         "Select sepal width (in cm)",
#         0.0, 5.0)

#     petalLength = st.slider(
#         "Select petal length (in cm)",
#         0.0, 3.0)

#     petalWidth = st.slider(
#         "Select petal width (in cm)",
#         0.0, 1.0)

#     if st.button("Indentify Flower"):
#         # turn input data into tensor
#         pred_data = np.array([sepalLength, sepalWidth, petalLength, petalWidth]).reshape(1,-1)
#         pred_data = scaler.transform(pred_data)
#         pred_data = torch.FloatTensor(pred_data)
#         pred_result = majority_vote(pred_data)
#         st.write(label_mapping[pred_result])


#     if st.button("Retrain network"):
#         train_network(model,optimizer,criterion,X_train,y_train,X_test,y_test,num_epochs,train_losses,test_losses)


#     if st.button("Save network"):  
#         save_network()

#     if st.button("Load network"):  
#         today = date.today()
#         year = today.year
#         month = today.month
#         day = today.day
#         dateString = f"{year:04d}{month:02d}{day:02d}"
#         path = "M:/People/James/test/python/FYP_stuff/models/"+dateString+"model.pth"
#         model = load_network(path)




def main():
    draw()

main()











#below is for testing network
# plt.figure(figsize=(10,10))
# plt.plot(train_losses, label='train loss')
# plt.plot(test_losses, label='test loss')
# plt.legend()
# plt.show()
# predictions_train = []
# predictions_test =  []
# with torch.no_grad():
#     predictions_train = model(X_train)
#     predictions_test = model(X_test)
# # Check how the predicted outputs look like and after taking argmax compare with y_train or y_test 
# #predictions_train  
# #y_train,y_test
# def get_accuracy_multiclass(pred_arr,original_arr):
#     if len(pred_arr)!=len(original_arr):
#         return False
#     pred_arr = pred_arr.numpy()
#     original_arr = original_arr.numpy()
#     final_pred= []
#     # we will get something like this in the pred_arr [32.1680,12.9350,-58.4877]
#     # so will be taking the index of that argument which has the highest value here 32.1680 which corresponds to 0th index
#     for i in range(len(pred_arr)):
#         final_pred.append(np.argmax(pred_arr[i]))
#     final_pred = np.array(final_pred)
#     count = 0
#     #here we are doing a simple comparison between the predicted_arr and the original_arr to get the final accuracy
#     for i in range(len(original_arr)):
#         if final_pred[i] == original_arr[i]:
#             count+=1
#     return count/len(final_pred)
# train_acc = get_accuracy_multiclass(predictions_train,y_train)
# test_acc  = get_accuracy_multiclass(predictions_test,y_test)
# print(f"Training Accuracy: {round(train_acc*100,3)}")
# print(f"Test Accuracy: {round(test_acc*100,3)}")


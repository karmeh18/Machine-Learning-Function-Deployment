import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
le=LabelEncoder()
scaler=StandardScaler()
norm=MinMaxScaler()
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_jobs=-1,oob_score=True)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(sampling_strategy='minority')
from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(sampling_strategy='majority')
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
gnb=GaussianNB()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import streamlit_pandas as sp
from PIL import Image


st.title("Machine Learning Function developed by Karan Mehta")
st.write('''In this website, I tried to combine Data Preprocessing like Standardization/Normalization with Label Encoder and Dummy variable added with the variety
         of different Machine Learning Algorithms, You just have to upload your file the code will automatically detect the presence of Missing Values and ask to 
         treat them afterwards you just have to select the technique with any Machine Learning Algorithms and as an output you will receive different accuracies with 
         respect to different threshold values and confusion matrix within which you can observe as to which threshold value with what accuracy is giving you are 
         lowest or the highest True Positive Rates/Fals Positive Rate or True Negative Rates/False Negative Rates.''')
image=Image.open("ML function flow chart.jpg")
st.write()
st.header("To check my LinkedIn Profile click on this link [LinkedIn Profile](https://www.linkedin.com/in/karan-mehta-313412162/)")
st.write()
st.header("To check my Git profile cling on the Link [Git-Hub Profile](https://github.com/karmeh18)")
st.header("Below is the flow chart of the function that is being executed in background")
st.image(image,caption="Machine Learning Function Flow Chart")
st.write()
st.write("Please click on the below buttons to download Demo datasets")
brain=pd.read_csv("brain_stroke.csv")
st.download_button(label="Download Brain Stroke File",data=brain.to_csv())

data_file=st.file_uploader("Upload Files",type=["csv"])
if data_file is not None:
    df=pd.read_csv(data_file)
    
#Deleting Desired Columns
delete=st.radio("Do you want to delete any columns ?",["Yes","No"])
if delete=="Yes":
    st.dataframe(df)
    col=st.multiselect("Please select the columns that you want to delete",df.columns)
    done=st.button("Do you want to delete the colums")
    if done==True:
        df.drop(columns=col,inplace=True)
    st.dataframe(df)
#Checking and filling the missing values in the data
object_columns=df.dtypes[df.dtypes==np.object].index
integer_columns=df.dtypes[(df.dtypes==np.int64) | (df.dtypes==np.float64)].index
if df.isnull().sum().sum()>0:
    missing=st.button("Do you want to see the missing values in your data")
    xx=pd.DataFrame(df[object_columns].isnull().sum()).reset_index().rename(columns={"index":"name",0:"Number of Missing values in Categorical Columns"})
    zxc=pd.DataFrame(df[integer_columns].isnull().sum()).reset_index().rename(columns={"index":"name",0:"Number of missing values in Numerical Columns"})
    if missing==True:
        st.table(xx)
        st.table(zxc)
    missing_button=st.button("Missing values have been identified in above columns, click to treat the missing values")
    if missing_button==True:
        MV_list=list(xx[xx["Number of Missing values in Categorical Columns"]>0]["name"].values)
        for i in MV_list:
            df[i]=df[i].fillna(df[i].value_counts().index[0])
        if df[integer_columns].isnull().sum().values.sum()>0:
            df.fillna(df.median(),inplace=True)
    st.table(df.isnull().sum())
    st.write("From the above it is clearly visible that the values have been replaced")
time_col=st.radio("Do you have date or time related column",["Yes","No"])
if time_col=="Yes":
    time=st.selectbox("Select the column",options=df.columns)
    df[time]=pd.to_datetime(df[time])
Dependent=st.selectbox("Please select the Dependent Variable from the Column's list",options=df.columns)
raw=st.radio("Do you want to transform the data or continue with RAW data",["Yes","No"])
if raw=="Yes":
    x1=st.radio("Do you want to Standardization or Normalization on Numerical Columns",["Standardization","Normalization"])
xx=st.radio("Do you want Dummy Variable or Label Encoder on Character Categorical Columns",["Dummy Variable","Label Encoder"])
X=df.drop(columns=[Dependent])
y=df[Dependent]
zz=df[Dependent].dtype==np.object
if zz==True:
    y=le.fit_transform(df[Dependent])
else:
         y=df[Dependent]


#Predict_threshold
def predict_threshold(model,X_test,threshold):
    import numpy as np
    return np.where(model.predict_proba(X_test)[:,1]>=threshold,1,0)


#ML Function
object_columns=X.dtypes[X.dtypes==np.object].index
integer_columns=X.dtypes[(X.dtypes==np.int64) | (X.dtypes==np.float64)].index
#Standardization, Normalization and RAW Data
if raw=="Yes":
    if x1=="Standardization":
        for i in integer_columns:
            X[i]=scaler.fit_transform(X[[i]])
    else:
        for i in integer_columns:
            X[i]=norm.fit_transform(X[[i]])
else:
    pass
    #Label Encoder and Dummy Variable
if xx=="Label Encoder":
    for i in object_columns:
        X[i]=le.fit_transform(X[i])
else:
    for ii in object_columns:
        x1=pd.get_dummies(X[ii],drop_first=True)
        X=pd.concat([X,x1],axis=1)
    X.drop(columns=object_columns,inplace=True)
#Treating Over Samples and Under Samples
first=pd.Series(y).value_counts().index[0]
first_iter=pd.Series(y).value_counts().values[0]
second=pd.Series(y).value_counts().index[1]
second_iter=pd.Series(y).value_counts().values[1]
if np.round(first_iter/y.shape[0],3)>=0.65:
         st.write()
         st.write('The iteration of {} category is {} that means the concentration of {} category on overall data is {}%'.format(first,first_iter,first,np.round((first_iter/y.shape[0])*100,3)))
         st.write("The iteration of {} category is {} that means the concentration of {} category on overall data is {}%".format(second,second_iter,second,np.round(second_iter/y.shape[0],3)*100))
         st.write()
         imbalance=st.selectbox("The target column is Imbalanced, for Stratified Random Sampling type YES else RUS OR ROS will gets activated!! ",options=["Yes","No"])
         if imbalance=="Yes":
                  X_train_reshape, X_test, y_train_reshape, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)
         else:
                  answer=st.radio("Do you want to continue with Random Over Sampling or Random Under Sampling please answer in ROS/ros or RUS/rus---- ",["ROS","RUS"])
                  if answer=="ROS" or answer=="ros" or answer=="Ros":
                           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                           X_train_reshape,y_train_reshape=ros.fit_resample(X_train,y_train)
                  else:
                           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                           X_train_reshape,y_train_reshape=rus.fit_resample(X_train,y_train)
else:
    X_train_reshape, X_test, y_train_reshape, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
ML_option=st.selectbox("Which Algorithm you want to use?",options=["Logistic Regression","Random Forest Classifier","Naive Bayes"])
if ML_option=="Logistic Regression":
    #Logistic Regression
    log.fit(X_train_reshape,y_train_reshape)
    log_pred=log.predict(X_test)
    log_accuracy=accuracy_score(y_test,log_pred)
    st.write()
    st.write("Accuracy of Logistic Regression on overall data is coming around at ",np.round(log_accuracy,3))
    st.write()
    for i in np.round(np.arange(0,1,0.1),1):
        y_predict=predict_threshold(log,X_test,i)
        st.write("Threshold: ",np.round(i,2))
        st.write(confusion_matrix(y_test,y_predict))
        st.write("Accuracy Score: with ",np.round(i,2)," threshold is ",np.round(accuracy_score(y_test,y_predict),3),"\n")
elif ML_option=="Random Forest Classifier":  
    #RandomForestClassification
    rfc.fit(X_train_reshape,y_train_reshape)
    rfc_pred=rfc.predict(X_test)
    rfc_accuracy=accuracy_score(y_test,rfc_pred)
    print()
    st.write("Accuracy of RandomForestClassification on overall data is coming around at ",np.round(rfc_accuracy,3))
    st.write()
    st.write('''The Meaning of OOB score is that in Random Forest Classifier when the algorithm is getting trained on different subsets of data, approximately 
    1/3rd datapoints from training subsets are unutilized. OOB Scores parameter tests the accuracy on these unutilized datapoints and can be considered as 
    validation data points to better understand the efficiency of Random Forest Classifier''')
    st.write("Accuracy from OOB Score is: ",np.round(rfc.oob_score_,3))
    st.write()
    for i in np.round(np.arange(0,1,0.1),1):
        y_predict=predict_threshold(rfc,X_test,i)
        st.write("Threshold: ",np.round(i,2))
        st.write(confusion_matrix(y_test,y_predict))
        st.write("Accuracy Score: with ",np.round(i,2)," threshold is ",np.round(accuracy_score(y_test,y_predict),3),"\n")
else:
    #NaiveBayes Classification
    gnb.fit(X_train_reshape,y_train_reshape)
    gnb_pred=gnb.predict(X_test)
    gnb_accuracy=accuracy_score(y_test,gnb_pred)
    st.write()
    st.write("Accuracy of NaiveBayesClassification on overall data is coming at ",np.round(gnb_accuracy,3))
    st.write()
    for i in np.round(np.arange(0,1,0.1),1):
        y_predict=predict_threshold(gnb,X_test,i)
        st.write("Threshold: ",np.round(i,2))
        st.write(confusion_matrix(y_test,y_predict))
        st.write("Accuracy Score: with ",np.round(i,2)," threshold is ",np.round(accuracy_score(y_test,y_predict),3),"\n")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import pickle
import time

randomseed=1

#Algorithms used in Classification
algorithms=("Decision_tree ","Adaboost_Decision_tree","SVC","Adaboost_SVC","Random_Forest","Adaboost_Random_Forest","KNN","Neive_Baies","Logistic_Regression")



##############

clfs_train_time_list = []
clfs_test_time_list = []
accuracy=[]
acuracylist =[]

def Training_Data():
    # reading training data
    dataset = pd.read_csv('heart_train.csv')

    names = dataset.columns[:13]
    X = dataset.iloc[:, :-1]
    X = X.loc[:, ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13']]
    print("xshape", X.shape)
    # split the data
    X_train = X.iloc[:, :13].to_numpy()
    y = dataset.iloc[:, -1].to_numpy()

    # random seed
    randomseed = 1

    # split the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=randomseed)

    # scaling the data
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # building the Decision tree model
    clf = tree.DecisionTreeClassifier(max_depth=3, random_state=randomseed)
    clf.fit(X_train, y_train)

    # calculating accuracy of Decision tree
    y_prediction = clf.predict(X_test)
    # print(y_prediction)

    tree_accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Decision Tree is " + str(tree_accuracy))
    accuracy.append(tree_accuracy)

    # building the Adaboost model
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, random_state=randomseed),
                             algorithm="SAMME",
                             n_estimators=100, random_state=randomseed)
    bdt.fit(X_train, y_train)

    adaboost_tree_accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Decision Tree and adaboost is " + str(adaboost_tree_accuracy))
    accuracy.append(adaboost_tree_accuracy)

    #########################SVC########################
    Clf_Svc = SVC(kernel='linear', C=1)
    Clf_Svc.fit(X_train, y_train)
    y_prediction = Clf_Svc.predict(X_test)

    svc_accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using SVM is " + str(svc_accuracy))
    accuracy.append(svc_accuracy)
    # building the Adaboost model
    bdt_Svc = AdaBoostClassifier(SVC(kernel='linear', C=1), learning_rate=0.01,
                                 algorithm="SAMME",
                                 n_estimators=100, random_state=randomseed)
    bdt_Svc.fit(X_train, y_train)

    # calculating accuracy of SVC
    y_prediction = bdt_Svc.predict(X_test)
    # print(y_prediction)

    adaboost_svc_accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using SVC and adapoost is " + str(adaboost_svc_accuracy))
    accuracy.append(adaboost_svc_accuracy)

    ######################################## RANDOM FOREST ##############################

    clf_ = RandomForestClassifier(n_estimators=120, max_depth=3, random_state=randomseed)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf_.fit(X_train, y_train)

    y_prediction = clf_.predict(X_test)

    random_forest_accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Random forest is " + str(random_forest_accuracy))
    accuracy.append(random_forest_accuracy)

    # building the Adaboost model
    bdt_R = AdaBoostClassifier(RandomForestClassifier(n_estimators=100, random_state=randomseed), learning_rate=0.01,
                               algorithm="SAMME",
                               n_estimators=100, random_state=randomseed)
    bdt_R.fit(X_train, y_train)

    # calculating accuracy of Decision tree
    y_prediction = bdt_R.predict(X_test)
    Adaboost_random_forest_accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using random forest and Adaboost is " + str(Adaboost_random_forest_accuracy))
    accuracy.append(Adaboost_random_forest_accuracy)

    ##################################### KNN #############################################

    knn = KNeighborsClassifier(n_neighbors=6)

    # Train the model using the training sets
    knn.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(X_test)
    Knn_accuracy = np.mean(y_pred == y_test) * 100
    print("The achieved accuracy using KNN is " + str(Knn_accuracy))
    accuracy.append(Knn_accuracy)

    #######################################NAIEVEBaise#############################
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    y_prediction = nb_clf.predict(X_test)
    NB_accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Naive Bayes classifier is " + str(NB_accuracy))
    accuracy.append(NB_accuracy)
    ####################### Logestic Regression ###################
    log_reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1,
                                 class_weight=None, random_state=randomseed)

    log_reg.fit(X_train, y_train)
    y_prediction = log_reg.predict(X_test)
    LG_accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Logestic Regression classifier is ", LG_accuracy)
    accuracy.append(LG_accuracy)

def Testing_Data():
    ############################# Reading Training Data#################################
        dataset = pd.read_csv('heart_train.csv')
        names = dataset.columns[:13]
        X = dataset.iloc[:, :-1]
        X = X.loc[:, ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13']]

        #print("xshape", X.shape)
        # split the data
        X_train = X.iloc[:, :13].to_numpy()
        y_train = dataset.iloc[:, -1].to_numpy()

        # random seed
        randomseed = 1

    ############################# Reading Testing Data#################################
        X_test = pd.read_csv('heart_test.csv')

        X_test = X_test.loc[:, ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13']]

        # scaling the data
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # building the Decision tree model
        clf = tree.DecisionTreeClassifier(max_depth=3,random_state=randomseed)

        #train time
        tree_train_time1 = time.time()

        clf.fit(X_train,y_train)

        tree_train_time2 = time.time()

        calc = tree_train_time2 - tree_train_time1
        clfs_train_time_list.append(calc)

        # plotting the tree
        #tree.plot_tree(clf.fit(X_train, y_train))
        #plt.show()

        # calculating accuracy of Decision tree
        #test time
        tree_test_time1 = time.time()
        y_prediction = clf.predict(X_test)
        tree_test_time2 = time.time()

        calc = tree_test_time2 - tree_test_time2
        clfs_test_time_list.append(calc)

        #accuracy=np.mean(y_prediction == y_test)*100
        #print ("The achieved accuracy using Decision Tree is " + str(accuracy))


        # building the Adaboost model

        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3,random_state=randomseed),
                                 algorithm="SAMME",
                                 n_estimators=100,random_state=randomseed)

        tree_ada1 = time.time()
        bdt.fit(X_train,y_train)

        tree_ada2 = time.time()

        calc = tree_ada2 - tree_ada1
        clfs_train_time_list.append(calc)

        ada_tree_test1 = time.time()
        y_prediction = bdt.predict(X_test)
        ada_tree_test2 = time.time()

        calc = ada_tree_test2 - ada_tree_test1
        clfs_test_time_list.append(calc)
        #########################SVC########################

        Clf_Svc = SVC(kernel='linear',C=1)

        svc_train_time1 = time.time()
        Clf_Svc.fit(X_train,y_train)
        svc_train_time2 = time.time()

        #append in train
        calc = svc_train_time2 - svc_train_time1
        clfs_train_time_list.append(calc)

        svc_test_time1 = time.time()
        y_prediction = Clf_Svc.predict(X_test)
        svc_test_time2 = time.time()

        #append in test
        calc = svc_test_time2 - svc_test_time1
        clfs_test_time_list.append(calc)

        #accuracy=np.mean(y_prediction == y_test)*100
        #print ("The achieved accuracy using SVM is " + str(accuracy))

        # building the Adaboost model


        bdt_Svc = AdaBoostClassifier(SVC(kernel='linear',C=1),learning_rate= 0.01,
                                 algorithm="SAMME",
                                 n_estimators=100,random_state=randomseed)

        svc_ada1 = time.time()
        bdt_Svc.fit(X_train,y_train)
        svc_ada2 = time.time()

        calc = svc_ada2 - svc_ada1
        clfs_train_time_list.append(calc)

        # calculating accuracy of Decision tree
        ada_svc_test1 = time.time()
        y_prediction = bdt_Svc.predict(X_test)
        ada_svc_test2 = time.time()

        calc = ada_svc_test2 - ada_svc_test1
        clfs_test_time_list.append(calc)

        #accuracy=np.mean(y_prediction == y_test)*100
        #print ("The achieved accuracy using SVM and adapoost is " + str(accuracy))


        ######################################## RANDOM FOREST ##############################

        clf_= RandomForestClassifier(n_estimators=120, max_depth=3,random_state=randomseed)

        #Train the model using the training sets y_pred=clf.predict(X_test)

        rf_train1 = time.time()
        clf_.fit(X_train,y_train)
        rf_train2 = time.time()

        calc = rf_train2 - rf_train1
        clfs_train_time_list.append(calc)

        rf_test1 = time.time()
        y_prediction=clf_.predict(X_test)
        rf_test2 = time.time()

        calc = rf_test2 - rf_test2
        clfs_test_time_list.append(calc)

        #accuracy=np.mean(y_prediction == y_test)*100
        #print ("The achieved accuracy using Random forest is " + str(accuracy))


        # building the Adaboost model
        bdt_R = AdaBoostClassifier(RandomForestClassifier(n_estimators=100,random_state=randomseed),learning_rate= 0.01,
                                 algorithm="SAMME",
                                 n_estimators=100,random_state=randomseed)

        rf_ada_train1 = time.time()
        bdt_R.fit(X_train,y_train)
        rf_ada_train2 = time.time()

        calc = rf_ada_train2 - rf_ada_train1
        clfs_train_time_list.append(calc)

        # calculating accuracy of Decision tree

        rf_ada_test1 = time.time()
        y_prediction = bdt_R.predict(X_test)
        rf_ada_test2 = time.time()

        calc = rf_ada_test2 - rf_ada_test1
        clfs_test_time_list.append(calc)

        #accuracy=np.mean(y_prediction == y_test)*100
        #print ("The achieved accuracy using Adaboost is " + str(accuracy))

        ##################################### KNN #############################################

        knn = KNeighborsClassifier(n_neighbors=6)

        #Train the model using the training sets
        knn_train1 = time.time()
        knn.fit(X_train, y_train)
        knn_train2 = time.time()

        calc = knn_train2 - knn_train1
        clfs_train_time_list.append(calc)

        #Predict the response for test dataset

        knn_test1 = time.time()
        y_pred = knn.predict(X_test)
        knn_test2 = time.time()

        calc = knn_test2 - knn_test1
        clfs_test_time_list.append(calc)

        #accuracy=np.mean(y_pred == y_test)*100
        #print ("The achieved accuracy using KNN is " + str(accuracy))

        #######################################NAIEVEbaise#############################

        nb_clf = GaussianNB()

        nb_train1 = time.time()
        nb_clf.fit(X_train, y_train)
        nb_train2 = time.time()

        calc = nb_train2 - nb_train1
        clfs_train_time_list.append(calc)

        nb_test1 = time.time()
        y_prediction = nb_clf.predict(X_test)
        nb_test2 = time.time()

        calc = nb_train2 - nb_train1
        clfs_test_time_list.append(calc)

        #NB_accuracy=np.mean(y_prediction == y_test)*100
        #print ("The achieved accuracy using Naive Bayes classifier is " + str(NB_accuracy))

        ##########################################
        log_reg=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.01, fit_intercept=True, intercept_scaling=1,
                           class_weight=None, random_state=randomseed)

        log_train1 = time.time()
        log_reg.fit(X_train, y_train)
        log_train2 = time.time()

        calc = log_train2 - log_train1
        clfs_train_time_list.append(calc)

        log_test1 = time.time()
        y_prediction = log_reg.predict(X_test)
        log_test2 = time.time()

        calc = log_test2 - log_test1
        clfs_test_time_list.append(calc)

        #LG_accuracy=np.mean(y_prediction == y_test)*100
        #print(LG_accuracy)
        return

def Get_Accuracy():

    pltacc=plt
    pltacc.ylabel('Accuracy')
    pltacc.title('Algorithms used with its accuracy')
    y_pos = np.arange(len(algorithms))
    pltacc.bar(y_pos, accuracy, color='darkorange', edgecolor='black')

    # Rotation of the bars names
    pltacc.xticks(y_pos, algorithms, fontsize=5.5, rotation=30)

    # Custom the subplot layout
    pltacc.subplots_adjust(bottom=0.4, top=0.99)
    pltacc.title(" Classifiers accuracy before PCA")
    # Show graphic
    pltacc.show()


def Training_time_graph():

    y_pos = np.arange(len(algorithms))
    plt1 = plt
    index = np.arange(len(algorithms))
    plt1.bar(index, clfs_train_time_list,color='darkorange',  edgecolor='black')
    plt1.xlabel('Classifiers', fontsize=10)
    plt1.ylabel('Training time', fontsize=10)
    plt1.xticks(index, algorithms, fontsize=5, rotation=30)
    plt1.title('Classifiers Training Time')
    plt1.show()


def Testing_time_graph():

    plt3=plt
    index = np.arange(len(algorithms))
    plt3.bar(index, clfs_test_time_list,color='darkorange',  edgecolor='black')
    plt3.xlabel('Classifiers', fontsize=10)
    plt3.ylabel('Testing time', fontsize=10)
    plt3.xticks(index, algorithms, fontsize=5, rotation=30)
    plt3.title('Classifiers Testing Time')
    plt3.show()

# All Calasifiers with PCA
def Training_Data_with_PCA():
    accuracy=[]
    clfs_test_time_list=[]
    clfs_train_time_list=[]
    dataset = pd.read_csv('heart_train.csv')

    names = dataset.columns[1:13]
    X = dataset.iloc[:, :-1]
    X = X.loc[:, ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13']]
    c = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13']
    print("xshape", X.shape)
    # split the data
    X = X.iloc[:, :13].to_numpy()

    #print("xshape 2", X.shape)

    scalar = StandardScaler()
    # # fitting
    scalar.fit(X)
    X = scalar.transform(X)
    # Importing PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=0.80)
    # pca = PCA()
    pca.fit(X)
    X_pca = pca.transform(X)

    #print(X_pca.shape)
    Pca_plot=plt
    Pca_plot.plot(np.cumsum(pca.explained_variance_ratio_))
    Pca_plot.xlabel("Number of components")
    Pca_plot.ylabel("Cumulative explained variance")
    Pca_plot.title("explained variance ratio of PCA")
    Pca_plot.show()

    #################################building data################333

    y = dataset.iloc[:, -1].to_numpy()

    # print shapes
    #print(y.shape)
    #print(y[:13])

    randomseed = 1

    # split the train and test data
    X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.4, random_state=randomseed)

    # building the Decision tree model
    clf = tree.DecisionTreeClassifier(max_depth=5, random_state=randomseed)
    clf.fit(X_pca_train, y_train)


    # calculating accuracy of Decision tree
    y_prediction = clf.predict(X_pca_test)
    accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Decision Tree  with PCA is " + str(accuracy))

    # building the Adaboost model
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, random_state=randomseed),
                             algorithm="SAMME",
                             n_estimators=100, random_state=randomseed)
    bdt.fit(X_pca_train, y_train)

    # calculating accuracy of Decision tree
    y_prediction = bdt.predict(X_pca_test)
    accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Adaboost  with PCA is " + str(accuracy))
    #########################SVC########################
    Clf_Svc = SVC(kernel='linear', C=0.5)
    Clf_Svc.fit(X_pca_train, y_train)
    y_prediction = Clf_Svc.predict(X_pca_test)
    accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using svc is " + str(accuracy))

    # building the Adaboost model
    bdt_Svc = AdaBoostClassifier(SVC(kernel='linear', C=0.5), learning_rate=0.1,
                                 algorithm="SAMME",
                                 n_estimators=50, random_state=randomseed)
    bdt_Svc.fit(X_pca_train, y_train)

    # calculating accuracy of Decision tree
    y_prediction = bdt_Svc.predict(X_pca_test)
    accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Adaboost  with PCA is " + str(accuracy))

    exit()
    ######################################## RANDOM FOREST ##############################

    clf_ = RandomForestClassifier(n_estimators=120, max_depth=3, random_state=randomseed)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf_.fit(X_pca_train, y_train)

    y_prediction = clf_.predict(X_pca_test)

    accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Random forest  with PCA is " + str(accuracy))

    # building the Adaboost model
    bdt_R = AdaBoostClassifier(RandomForestClassifier(n_estimators=120, max_depth=3, random_state=randomseed),
                               learning_rate=0.01,
                               algorithm="SAMME",
                               n_estimators=100, random_state=randomseed)
    bdt_R.fit(X_pca_train, y_train)

    # calculating accuracy of Decision tree
    y_prediction = bdt_R.predict(X_pca_test)
    accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Adaboost  with PCA is " + str(accuracy))

    ##################################### KNN #############################################

    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the model using the training sets
    knn.fit(X_pca_train, y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(X_pca_test)
    accuracy = np.mean(y_pred == y_test) * 100
    print("The achieved accuracy using KNN  with PCA is " + str(accuracy))

    #######################################NAIEVEbaise#############################
    nb_clf = GaussianNB()
    nb_clf.fit(X_pca_train, y_train)
    y_prediction = nb_clf.predict(X_pca_test)
    NB_accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Naive Bayes classifier  with PCA is " + str(NB_accuracy))

    ##########################################
    log_reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.07, fit_intercept=True, intercept_scaling=1,
                                 class_weight=None, random_state=randomseed)

    log_reg.fit(X_pca_train, y_train)
    y_prediction = log_reg.predict(X_pca_test)
    LG_accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Logistic Regression with PCA  is: ",LG_accuracy)

    return

#######################################################################################
#Best three Classifiers
def Best_Classifiers_with_PCA():

    clfs_train_time_list = []
    clfs_test_time_list = []
    acuracylist = []
    algorithms = ["SVC", "Random_Forest", "Logistic_Regression"]

    dataset = pd.read_csv('heart_train.csv')

    names = dataset.columns[1:13]
    X = dataset.iloc[:, :-1]
    X = X.loc[:, ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13']]
    c = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13']
    #print("xshape", X.shape)
    # split the data
    X = X.iloc[:, :13].to_numpy()

    #print("xshape 2", X.shape)

    scalar = StandardScaler()
    # # fitting
    scalar.fit(X)
    X = scalar.transform(X)
    # Importing PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=0.80)
    # pca = PCA()
    pca.fit(X)
    X_pca = pca.transform(X)

    filename = 'PCA.sav'
    pickle.dump(pca, open(filename, 'wb'))

    ################################# building data ################

    y = dataset.iloc[:, -1].to_numpy()
    # print shapes
    #print(y.shape)
    #print(y[:13])
    #randomseed = 1
    # split the train and test data
    X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.4, random_state=randomseed)
    # scaling the data
    # scaler = StandardScaler()
    # scaler.fit(X_pca_train)

    # X_pca_train = scaler.transform(X_pca_train)
    # X_pca_test = scaler.transform(X_pca_test)
    clfs_train_time_list = []
    clfs_test_time_list = []
    acuracylist = []
    algorithms = ["SVC", "Random_Forest", "Logistic_Regression"]

    #########################SVC########################
    Clf_Svc = SVC(kernel='linear', C=0.5)

    tree_train_time1 = time.time()
    Clf_Svc.fit(X_pca_train, y_train)

    tree_train_time2 = time.time()

    calc = tree_train_time2 - tree_train_time1
    clfs_train_time_list.append(calc)

    tree_train_time1 = time.time()
    y_prediction = Clf_Svc.predict(X_pca_test)
    tree_train_time2 = time.time()

    calc = tree_train_time2 - tree_train_time1
    clfs_test_time_list.append(calc)

    accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using svc is " + str(accuracy))
    acuracylist.append(accuracy)
    ######################################## RANDOM FOREST ##############################

    clf_ = RandomForestClassifier(n_estimators=120, max_depth=3, random_state=randomseed)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    tree_train_time1 = time.time()
    clf_.fit(X_pca_train, y_train)
    tree_train_time2 = time.time()

    calc = tree_train_time2 - tree_train_time1
    clfs_train_time_list.append(calc)

    tree_train_time1 = time.time()
    y_prediction = clf_.predict(X_pca_test)
    tree_train_time2 = time.time()

    calc = tree_train_time2 - tree_train_time1
    clfs_test_time_list.append(calc)

    accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Random forest is " + str(accuracy))
    acuracylist.append(accuracy)

    ##########################################
    log_reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.07, fit_intercept=True, intercept_scaling=1,
                                 class_weight=None, random_state=randomseed)
    tree_train_time1 = time.time()
    log_reg.fit(X_pca_train, y_train)
    tree_train_time2 = time.time()

    calc = tree_train_time2 - tree_train_time1
    clfs_train_time_list.append(calc)

    tree_train_time1 = time.time()
    y_prediction = log_reg.predict(X_pca_test)
    tree_train_time2 = time.time()
    calc = tree_train_time2 - tree_train_time1
    clfs_test_time_list.append(calc)

    LG_accuracy = np.mean(y_prediction == y_test) * 100
    print(LG_accuracy)
    acuracylist.append(LG_accuracy)

    return algorithms,acuracylist,clfs_train_time_list,clfs_test_time_list


def Training_time_with_PCA_graph(algorithms,clfs_train_time_list):

    train = plt
    index = np.arange(len(algorithms))
    train.bar(index, clfs_train_time_list,color='red', edgecolor='black')
    train.xlabel('Classifiers')
    train.ylabel('Training time after using PCA')
    train.xticks(index, algorithms, rotation=20)
    train.title('Classifiers Training Time')
    train.ylim(ymax=0.07, ymin=0.0)
    train.show()

def Testing_time_with_PCA(algorithms,clfs_test_time_list):
    test = plt
    index = np.arange(len(algorithms))
    test.bar(index, clfs_test_time_list,color='red', edgecolor='black')
    test.xlabel('Classifiers', fontsize=5)
    test.ylabel('Testing time')
    test.xticks(index, algorithms, rotation=20)
    test.title('Classifiers Testing Time')
    test.ylim(ymax=0.2, ymin=0.0)
    test.show()

def Accuracy_Graph_after_PCA(algorithms,acuracylist):

    acc = plt
    index = np.arange(len(algorithms))
    acc.bar(index, acuracylist,color='red', edgecolor='black')
    acc.xlabel('Classifiers')
    acc.ylabel('Accuracy')
    acc.xticks(index, algorithms, rotation=20)
    acc.title('Classifiers accuracy')
    acc.show()

print(" Starting Spliting Training The Data")
Training_Data()


print(" Starting Get The accuracy of Training The Data")

Get_Accuracy()

print(" Starting Testing The Testing File")
Testing_Data()

print( "Training Time", clfs_train_time_list)
print( "Testing Time", clfs_test_time_list)

Training_time_graph()

Testing_time_graph()
################## PCA ######################
Training_Data_with_PCA()


algorithms,acuracylist,clfs_train_time_list,clfs_test_time_list=Best_Classifiers_with_PCA()


print("testing after PCA", clfs_test_time_list)
print("training after PCA", clfs_train_time_list)

Training_time_with_PCA_graph(algorithms,clfs_train_time_list)
Testing_time_with_PCA(algorithms,clfs_test_time_list)


Accuracy_Graph_after_PCA(algorithms,acuracylist)
print("accuracy after PCA ", accuracy)

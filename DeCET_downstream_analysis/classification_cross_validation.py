import sys
import numpy as np
from collections import defaultdict
from collections import Counter
import time
import tensorly
import torch
import argparse
from sklearn import svm
import importlib

def l1_scaling(X):
    """l1 normalize along the last index.
    
    Fix all but the last index and scale the resulting
    vector to unit l1 norm. Can be used to scale sequencing
    data to the same read depth.
    """
    Xn = torch.norm(X, p=1, dim=-1, keepdim=True)
    return X.div(Xn)

def full_tensor_product(X,U):
    """Compute the n-mode product with the list of matrices.

    This function takes a tensor X and a list of matrices U
    for each vector space and computes the corresponding 
    n-mode product of the tensor with each of the matrices
    X x_1 U^(1) x_2 U^(2)...x_p U^(p)
    Be careful about using the correct transposition for the
    context.
    """
    p = len(U)
    s = tensorly.tenalg.mode_dot(X, U[0],0)
    for i in range(1,p):
        s = tensorly.tenalg.mode_dot(s,U[i],i)
    return s

def HOSVD(X):
    """Compute the higher order singular value decomposition of X.

    Given a tensor X, obtain the HOSVD of the tensor
    and return the orthogonal matrix U for each subspace.
    Note that this function utilizes rank restrictions to
    reduce the SVD computation when applicable.  
    """
    U = []
    for i in range(0,len(dim)):
        X_n = tensorly.base.unfold(X,i)
        l = X_n.size()
        print(i)
        print( time.asctime( time.localtime(time.time()) ))
        sys.stdout.flush()
        if(l[0] <= l[1]):
            T = torch.mm(X_n,torch.t(X_n))
            E = torch.symeig(T,eigenvectors=True)
            eig_sort = torch.sort(E[0],0, descending=True)
            U.append(E[1][:,eig_sort[1]])
        else:
            T = torch.mm(torch.t(X_n),X_n)
            E = torch.symeig(T,eigenvectors=True)
            eig_sort = torch.sort(E[0],0, descending=True)
            V = torch.mm(X_n,E[1][:,eig_sort[1]])
            # Remove eigenvectors with zero eigenvalue
            # to prevent division by zero.
            nzi = torch.nonzero(eig_sort[0])
            U.append(V[:,nzi[:,0]]/torch.sqrt(eig_sort[0][nzi[:,0]]))
    S = full_tensor_product(X,[torch.t(U[i]) for i in range(0,len(U))])
    return U, S

## STM functions (writen by Miroslav Hejna) ##
def T_Product(m_X,m_w0,m_w1):
    T_Prod=np.dot(np.dot(m_X,m_w1),m_w0);
    return T_Prod;

def PredictValue(m_X,m_w0,m_w1,m_b):
    T_Predval=np.dot(np.dot(m_X,m_w1),m_w0)+m_b;
    return T_Predval;

def Predict(m_X,m_w0,m_w1,m_b):
    T_predval=np.dot(np.dot(m_X,m_w1),m_w0)+m_b;
    T_Pred=-1;
    if (T_predval>=0): T_Pred=1;
    return T_Pred;
    
def Contstraint(m_X,m_w0,m_w1,m_b):    
    T_Contstraint=np.dot(np.dot(m_X,m_w1),m_w0)+m_b;
    return T_Contstraint;

def LossFunction(X_vec,Y_vec,m_w0,m_w1,m_b,m_lambda1,m_lambda2):    
    LF=0.5*m_lambda1*np.dot(m_w0,m_w0)*np.dot(m_w1,m_w1)+0.25*m_lambda2*np.dot(m_w0,m_w0)*np.dot(m_w0,m_w0)+0.25*m_lambda2*np.dot(m_w1,m_w1)*np.dot(m_w1,m_w1);
    for i in range(len(X_vec)): LF=LF+max(0,-(Y_vec[i]*Contstraint(X_vec[i],m_w0,m_w1,m_b)-1));
    return LF;

def TrainSTM(alpha,Lambda1,Lambda2,X_T,Y_T,w0_I,w1_I,b_I):

    #initialize Parametres    
    [Fixed_w0_Flag,Fixed_w1_Flag]=[False,False];
    [w0,w1,b]=[w0_I,w1_I,b_I];
    [X,Y]=[X_T,Y_T];


    N_Epochs=400; #number of epochs
    N_SubEpochs=250; #number of epochs
    
    if (Fixed_w0_Flag): w0=np.array([1.0,0,0]);
    if (Fixed_w1_Flag): w1=np.array([1.0,0,0,0,0,0,0,0,0,0]);

    AccuracyDict=defaultdict(lambda:0,Counter([Y[i]==Predict(X[i],w0,w1,b) for i in range(len(X))]));
    Accuracy=AccuracyDict[True]/float(AccuracyDict[True]+AccuracyDict[False]);
    [w0_history,w1_history,b_history,LF_history,Accuracy_history]=[[w0],[w1],[b],[LossFunction(X,Y,w0,w1,b,Lambda1,Lambda2)],[Accuracy]];
    for n in range(N_Epochs):
        #print(str(n))

        #update rule
        #do minimization in turn
        if (not Fixed_w0_Flag):
            for n_s in range(N_SubEpochs):
                for i in range(len(X)): 
                    if(Y[i]*Contstraint(X[i],w0,w1,b)<1): w0=w0+alpha*(Y[i]*np.dot(X[i],w1)-Lambda1*np.dot(w1,w1)*w0-Lambda2*np.dot(w0,w0)*w0);
                    else: w0=w0-alpha*(Lambda1*np.dot(w1,w1)*w0+Lambda2*np.dot(w0,w0)*w0);   
                for i in range(len(X)):
                    if(Y[i]*Contstraint(X[i],w0,w1,b)<1): b=b+alpha*(Y[i]); 
                    else: continue;
        
        if (not Fixed_w1_Flag):    
            for n_s in range(N_SubEpochs):              
                for i in range(len(X)): 
                    if(Y[i]*Contstraint(X[i],w0,w1,b)<1): w1=w1+alpha*(Y[i]*np.dot(w0,X[i])-Lambda1*np.dot(w0,w0)*w1-Lambda2*np.dot(w1,w1)*w1);
                    else: w1=w1-alpha*(Lambda1*np.dot(w0,w0)*w1+Lambda2*np.dot(w1,w1)*w1);   
                for i in range(len(X)):
                    if(Y[i]*Contstraint(X[i],w0,w1,b)<1): b=b+alpha*(Y[i]); 
                    else: continue;
    
    
        AccuracyDict=defaultdict(lambda:0,Counter([Y[i]==Predict(X[i],w0,w1,b) for i in range(len(X))]));
        Accuracy=AccuracyDict[True]/float(AccuracyDict[True]+AccuracyDict[False]);
        [w0_history.append(w0),w1_history.append(w1),b_history.append(b),LF_history.append(LossFunction(X,Y,w0,w1,b,Lambda1,Lambda2)),Accuracy_history.append(Accuracy)];
        #print(str(np.linalg.norm(w0-w0_history[n]))+'\t'+str(np.linalg.norm(w1-w1_history[n]))+'\t'+str(np.abs(b-b_history[n]))+'\t'+str(LossFunction(X,Y,w0,w1,b,Lambda1,Lambda2)));  

    return [w0_history,w1_history,b_history,LF_history,Accuracy_history]; 

if __name__ == '__main__':

    #######
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--data_tensor", type=str, required=True,
                        help=("The data tensor for the 21 patient matched leiomyoma \n"
                              "and myometrium datasets. This script uses the        \n"
                              "specific patient ordering for assigning class labels,\n"
                              "and so should not be used for different data tensors.\n"
                              "Required."))
    parser.add_argument("-a", "--addition_test", type=str, required=True,
                        help=("Additional patient data tensor. As for the data      \n"
                              "tensor, the sample ordering is used for assigning    \n"
                              "class labels. Required.\n"))
    parser.add_argument("--scale_function", nargs='?', const="l1_scaling", default=None,
                        help=("Specify a method for scaling the input data tensor. \n"
                              "If this argument is given with no value, the column \n"
                              "vectors obtained from fixing all but the last index \n"
                              "will be scaled to unit l1 norm. Otherwise the name  \n"
                              "of a module can be given. The module must specify a \n"
                              "function named scale_function that takes an input   \n"
                              "tensor and returns a scaled version of this tensor. \n"
                              "Default is to not scale the input tensor in any way"))
    parser.add_argument("--n_proj", type=int, required=True,
                   help=("The number of projections to use for classification"))

    args = parser.parse_args()

    n_proj = args.n_proj

    print( time.asctime( time.localtime(time.time()) ))

    t_test = torch.load(args.addition_test)

    t_full = torch.load(args.data_tensor)
    ###
    dim = tuple(t_full.shape)
    print(dim)
    sys.stdout.flush()
    ###
    
    n_iters = dim[1]
    n_tr = dim[1] - 1 

    # This script is highly specific to the dataset used in this
    # study. The mutation labels are manually specified here for
    # the leiomyoma samples. Ensure the input tensor follows the
    # sample ordering given in GSE142332. 
    # MED12 mutation labels
    MED12_list = [0, 2, 3, 4, 6, 10, 11, 12, 14, 15, 17, 18, 20]
    class_dic = {}
    for j in range(dim[1]):
        if(j in MED12_list):
            class_dic[j] = -1
        elif(j not in MED12_list):
            class_dic[j] = 1
    # Disease condition and MED12 mutation labels
    # of additional test samples
    Y_add_test = [1, 1, 1, 1, 1, 1, 1]
    Y_add_test_MED12 = [-1, 1, 1, 1, -1, -1, -1]

    tensorly.set_backend('pytorch')
    device = torch.device('cuda')

    t_test = t_test.to(device)

    ##################################
    accuracy_record_train = np.zeros((dim[2] + 2, n_iters))
    accuracy_record_test = np.zeros((dim[2] + 2, n_iters))
    accuracy_record_add_test = np.zeros((dim[2] + 2, n_iters))
    accuracy_record_train_MED12 = np.zeros((dim[2] + 2, n_iters))
    accuracy_record_test_MED12 = np.zeros((dim[2] + 2, n_iters))
    accuracy_record_add_test_MED12 = np.zeros((dim[2] + 2, n_iters))
    # Initialize the tensor
    for repeat in range(0,n_iters):
        print("Repeat: ", repeat)
        sys.stdout.flush()
        # Get the training set patient indices
        tr_set = []
        for i in range(dim[1]):
            if(i != repeat):
                tr_set.append(i)
            else:
                test_pt = i
        ##########################

        print(n_tr)
        t = torch.tensor(t_full[:,tr_set,:,:])
        if(args.scale_function==l1_scaling):
            print("Using l1")
            t = l1_scaling(t)
        elif(args.scale_function!=None):
            print("Using {} for scaling".format(args.scale_function))
            scale_module = importlib.import_module(args.scale_function)
            scale_function = scale_module.scale_function
            t = scale_function(t)

        print( 'moving to device' )
        print( time.asctime( time.localtime(time.time()) ))
        sys.stdout.flush()

        t = t.to(device)

        print( 'starting tensor decomposition' )
        print( time.asctime( time.localtime(time.time()) ))
        sys.stdout.flush()
        U, S = HOSVD(t)
        print( 'finished tensor decomposition' )
        print( time.asctime( time.localtime(time.time()) ))
        print( 'starting approximation algorithm' )
        sys.stdout.flush() 

        B = tensorly.tenalg.mode_dot(S, U[0], 0)
        B = tensorly.tenalg.mode_dot(B, U[1], 1)
        B = tensorly.tenalg.mode_dot(B, U[2], 2) 
        B = B[:,:,:,:n_proj]
        print(B.size())
        Bn = torch.norm(B, p=2, dim=-1, keepdim=True)
        print(Bn.size())
        norm_proj = B.div(Bn)
        print(norm_proj.size())
        del B

        X_STM = torch.reshape(norm_proj, (dim[0]*n_tr, dim[2], n_proj))
        X_MED12_STM = torch.reshape(norm_proj[1,:,:,:], (n_tr, dim[2], n_proj))

        X = torch.reshape(norm_proj, (dim[0]*n_tr, dim[2]*n_proj))
        X_MED12 = torch.reshape(norm_proj[1,:,:,:], (n_tr, dim[2]*n_proj)) 

        Y = np.concatenate((-1*np.ones(n_tr,dtype=int), np.ones(n_tr,dtype=int)))
        Y_MED12 = []
        for j, pt_j in enumerate(tr_set):
            Y_MED12.append(class_dic[pt_j])

        test_data = torch.tensor(t_full[:,test_pt,:,:])
        test_data = test_data.to(device)
        test_proj = torch.matmul(test_data, U[-1])
        del test_data
        test_proj = test_proj[:,:,:n_proj]
        tn = torch.norm(test_proj, p=2, dim=-1, keepdim=True)
        test_proj = test_proj.div(tn)
        X_test = torch.reshape(test_proj, (dim[0], dim[2]*n_proj))

        test_add_proj = torch.matmul(t_test, U[-1])
        test_add_proj = test_add_proj[:,:,:n_proj]
        tn = torch.norm(test_add_proj, p=2, dim=-1, keepdim=True)
        test_add_proj = test_add_proj.div(tn)
        X_add_test_STM = torch.reshape(test_add_proj, (test_add_proj.size()[0], dim[2], n_proj))
        X_add_test = torch.reshape(test_add_proj, (test_add_proj.size()[0], dim[2]*n_proj))

        print("tr_set:            ",tr_set)
        ############## STM classification ###
        # Set STM parameters
        alpha=0.0001; #learning rate
        Lambda2=0.0; #regularization
        Lambda1=0.0005; #regularization
        Scale=0.1

        # Initial STM parameters to be fit
        w0_init=Scale*np.random.rand(dim[2]);
        w1_init=Scale*np.random.rand(n_proj);
        b_init=Scale*np.random.rand(1)[0]
        # Fit STM parameters 
        [w0_history,w1_history,b_history,LF_history,Accuracy_history]=TrainSTM(alpha,Lambda1,Lambda2,X_STM.to('cpu'),Y,w0_init,w1_init,b_init);
        [w0,w1,b,LF,Accuracy]=[w0_history[-1],w1_history[-1],b_history[-1],LF_history[-1],Accuracy_history[-1]];

        clf = svm.LinearSVC(C=1, loss='squared_hinge', max_iter=10000 ,tol=1e-5, class_weight='balanced')
        clf.coef_ = np.reshape(np.outer(w0, w1).flatten(order='C'),(-1,dim[2]*n_proj))
        clf.intercept_ = [b]
        clf.classes_ = np.array([-1, 1])

        print("All histones disease STM classification")
        print("STM  predictions")
        print(clf.predict(X))
        print(clf.predict(X_test))
        print(clf.predict(X_add_test))

        print("STM accuracy")
        print(clf.score(X,Y))
        print(clf.score(X_test,np.array([-1,1])))
        print(clf.score(X_add_test,Y_add_test))

        accuracy_record_train[-1,repeat] = clf.score(X,Y)
        accuracy_record_test[-1,repeat] = clf.score(X_test,np.array([-1,1]))
        accuracy_record_add_test[-1,repeat] = clf.score(X_add_test,Y_add_test)

        # Initial STM parameters to be fit
        w0_init=Scale*np.random.rand(dim[2]);
        w1_init=Scale*np.random.rand(n_proj);
        b_init=Scale*np.random.rand(1)[0]
        # Fit STM parameters 
        [w0_history,w1_history,b_history,LF_history,Accuracy_history]=TrainSTM(alpha,Lambda1,Lambda2,X_MED12_STM.to('cpu'),Y_MED12,w0_init,w1_init,b_init);
        [w0,w1,b,LF,Accuracy]=[w0_history[-1],w1_history[-1],b_history[-1],LF_history[-1],Accuracy_history[-1]];

        clf.coef_ = np.reshape(np.outer(w0, w1).flatten(order='C'),(-1,dim[2]*n_proj))
        clf.intercept_ = [b]

        print("All histones MED12 STM classification")
        print("STM  predictions")
        print(clf.predict(X_MED12))
        print(clf.predict(X_test[1].reshape(1, -1)))
        print(clf.predict(X_add_test))

        print("STM accuracy")
        print(clf.score(X_MED12, Y_MED12))
        print(clf.score(X_test[1].reshape(1, -1), [class_dic[test_pt]]))
        print(clf.score(X_add_test, Y_add_test_MED12))

        accuracy_record_train_MED12[-1,repeat] = clf.score(X_MED12, Y_MED12)
        accuracy_record_test_MED12[-1,repeat] = clf.score(X_test[1].reshape(1, -1), [class_dic[test_pt]])
        accuracy_record_add_test_MED12[-1,repeat] = clf.score(X_add_test,Y_add_test_MED12)

        ############## SVM test #############
        clf = svm.LinearSVC(C=1, loss='squared_hinge', max_iter=10000 ,tol=1e-5, class_weight='balanced')
        clf.fit(X, Y)

        print("All histones disease classification")
        print("SVM  predictions")
        print(clf.predict(X))
        print(clf.predict(X_test))
        print(clf.predict(X_add_test))

        print("SVM accuracy")
        print(clf.score(X,Y))
        print(clf.score(X_test,np.array([-1,1])))
        print(clf.score(X_add_test,Y_add_test))

        accuracy_record_train[0,repeat] = clf.score(X,Y)
        accuracy_record_test[0,repeat] = clf.score(X_test,np.array([-1,1]))
        accuracy_record_add_test[0,repeat] = clf.score(X_add_test,Y_add_test)

        clf = svm.LinearSVC(C=1, loss='squared_hinge', max_iter=10000 ,tol=1e-5, class_weight='balanced')
        clf.fit(X_MED12, Y_MED12)

        ############## SVM test #############
        print("All histones MED12 classification")
        print("SVM  predictions")
        print(clf.predict(X_MED12))
        print(clf.predict(X_test[1].reshape(1, -1)))
        print(clf.predict(X_add_test))

        print("SVM accuracy")
        print(clf.score(X_MED12, Y_MED12))
        print(clf.score(X_test[1].reshape(1, -1), [class_dic[test_pt]]))
        print(clf.score(X_add_test, Y_add_test_MED12))

        accuracy_record_train_MED12[0,repeat] = clf.score(X_MED12, Y_MED12)
        accuracy_record_test_MED12[0,repeat] = clf.score(X_test[1].reshape(1, -1), [class_dic[test_pt]])
        accuracy_record_add_test_MED12[0,repeat] = clf.score(X_add_test,Y_add_test_MED12)


        for k in range(dim[2]):
            X = torch.reshape(norm_proj[:,:,k,:], (dim[0]*n_tr, n_proj))
            X_test = torch.reshape(test_proj[:,k,:], (dim[0], n_proj))
            X_add_test = torch.reshape(test_add_proj[:,k,:], (test_add_proj.size()[0], n_proj))
            clf.fit(X,Y)

            ############## SVM test #############
            print("Assay = ", k)
            print("SVM  predictions")
            print(clf.predict(X))
            print(clf.predict(X_test))
            print(clf.predict(X_add_test))

            print("SVM accuracy")
            print(clf.score(X,Y))
            print(clf.score(X_test,np.array([-1,1])))
            print(clf.score(X_add_test,Y_add_test))

            accuracy_record_train[k+1, repeat] = clf.score(X,Y)
            accuracy_record_test[k+1, repeat] = clf.score(X_test, np.array([-1,1]))
            accuracy_record_add_test[k+1, repeat] = clf.score(X_add_test,Y_add_test) 

            X_MED12 = torch.reshape(norm_proj[1,:,k,:], (n_tr, n_proj))
            clf.fit(X_MED12,Y_MED12)

            ############## SVM test #############
            print("MED12 Assay = ", k)
            print("SVM  predictions")
            print(clf.predict(X_MED12))
            print(clf.predict(X_test[1].reshape(1, -1)))
            print(clf.predict(X_add_test))

            print("SVM accuracy")
            print(clf.score(X_MED12,Y_MED12))
            print(clf.score(X_test[1].reshape(1, -1), [class_dic[test_pt]]))
            print(clf.score(X_add_test,Y_add_test_MED12))

            accuracy_record_train_MED12[k+1, repeat] = clf.score(X_MED12,Y_MED12)
            accuracy_record_test_MED12[k+1, repeat] = clf.score(X_test[1].reshape(1, -1), [class_dic[test_pt]])
            accuracy_record_add_test_MED12[k+1, repeat] = clf.score(X_add_test,Y_add_test_MED12)         

    ##########################################
    print("Mean STM accuracy all marks")
    print("Train: ", np.mean(accuracy_record_train[-1]))
    print("Test: ", np.mean(accuracy_record_test[-1]))
    print("Additional Test: ", np.mean(accuracy_record_add_test[-1]))

    print("Mean accuracy all marks")
    print("Train: ", np.mean(accuracy_record_train[0]))
    print("Test: ", np.mean(accuracy_record_test[0]))
    print("Additional Test: ", np.mean(accuracy_record_add_test[0]))

    print("Mean accuracy H3K27ac")
    print("Train: ", np.mean(accuracy_record_train[1]))
    print("Test: ", np.mean(accuracy_record_test[1]))
    print("Additional Test: ", np.mean(accuracy_record_add_test[1]))

    print("Mean accuracy H3K4me3")
    print("Train: ", np.mean(accuracy_record_train[2]))
    print("Test: ", np.mean(accuracy_record_test[2]))
    print("Additional Test: ", np.mean(accuracy_record_add_test[2]))

    print("Mean accuracy H3K4me1")
    print("Train: ", np.mean(accuracy_record_train[3]))
    print("Test: ", np.mean(accuracy_record_test[3]))
    print("Additional Test: ", np.mean(accuracy_record_add_test[3]))

    print("Mean MED12 STM accuracy all marks")
    print("Train: ", np.mean(accuracy_record_train_MED12[-1]))
    print("Test: ", np.mean(accuracy_record_test_MED12[-1]))
    print("Additional Test: ", np.mean(accuracy_record_add_test_MED12[-1]))

    print("Mean MED12 accuracy all marks")
    print("Train: ", np.mean(accuracy_record_train_MED12[0]))
    print("Test: ", np.mean(accuracy_record_test_MED12[0]))
    print("Additional Test: ", np.mean(accuracy_record_add_test_MED12[0]))

    print("Mean MED12 accuracy H3K27ac")
    print("Train: ", np.mean(accuracy_record_train_MED12[1]))
    print("Test: ", np.mean(accuracy_record_test_MED12[1]))
    print("Additional Test: ", np.mean(accuracy_record_add_test_MED12[1]))

    print("Mean MED12 accuracy H3K4me3")
    print("Train: ", np.mean(accuracy_record_train_MED12[2]))
    print("Test: ", np.mean(accuracy_record_test_MED12[2]))
    print("Additional Test: ", np.mean(accuracy_record_add_test_MED12[2]))

    print("Mean MED12 accuracy H3K4me1")
    print("Train: ", np.mean(accuracy_record_train_MED12[3]))
    print("Test: ", np.mean(accuracy_record_test_MED12[3]))
    print("Additional Test: ", np.mean(accuracy_record_add_test_MED12[3]))

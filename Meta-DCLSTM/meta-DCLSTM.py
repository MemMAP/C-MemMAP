import torch, os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import glob
import  numpy as np
import sys

PROJECT_ROOT_DIRECTORY = "./"
NP_TRACE_DIRECTORY ="./np_file/"
sys.path.append(PROJECT_ROOT_DIRECTORY)

from utils import *
from LSTM import LSTM
from Meta_memsys import META

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)


def load_dataset(data):
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    return X_train, X_test, y_train, y_test

file_list = sorted(glob.glob(NP_TRACE_DIRECTORY + '/*.npz'))


X0_train, X0_test, y0_train, y0_test = load_dataset(np.load(file_list[0]))
X1_train, X1_test, y1_train, y1_test = load_dataset(np.load(file_list[1]))
X2_train, X2_test, y2_train, y2_test = load_dataset(np.load(file_list[2]))
X3_train, X3_test, y3_train, y3_test = load_dataset(np.load(file_list[3]))
X4_train, X4_test, y4_train, y4_test = load_dataset(np.load(file_list[4]))
X5_train, X5_test, y5_train, y5_test = load_dataset(np.load(file_list[5]))
X6_train, X6_test, y6_train, y6_test = load_dataset(np.load(file_list[6]))
X7_train, X7_test, y7_train, y7_test = load_dataset(np.load(file_list[7]))
X8_train, X8_test, y8_train, y8_test = load_dataset(np.load(file_list[8]))
X9_train, X9_test, y9_train, y9_test = load_dataset(np.load(file_list[9]))
X10_train, X10_test, y10_train, y10_test = load_dataset(np.load(file_list[10]))
X11_train, X11_test, y11_train, y11_test = load_dataset(np.load(file_list[11]))
X12_train, X12_test, y12_train, y12_test = load_dataset(np.load(file_list[12]))

X_train_all = np.stack((X0_train,X1_train,X2_train,X3_train,X4_train,X5_train,X6_train,X7_train,X8_train,X9_train,X10_train,X11_train,X12_train),axis = 0)
X_test_all = np.stack((X0_test,X1_test,X2_test,X3_test,X4_test,X5_test,X6_test,X7_test,X8_test,X9_test,X10_test,X11_test,X12_test),axis = 0)
y_train_all = np.stack((y0_train,y1_train,y2_train,y3_train,y4_train,y5_train,y6_train,y7_train,y8_train,y9_train,y10_train,y11_train,y12_train),axis = 0)
y_test_all = np.stack((y0_test,y1_test,y2_test,y3_test,y4_test,y5_test,y6_test,y7_test,y8_test,y9_test,y10_test,y11_test,y12_test),axis = 0)

# n_way, k_shot: each batch n*k examples
class args:
    '''
    parameters
    '''
    epoch = 500
    #epoch = 2
    n_way = 1
    k_shot = 256 #each batch, from n class each sample k examples. total n*k
    k_spt = 256 #spt=shot
    k_qry = 1024
    task_num = 13
    meta_lr = 1e-3
    update_lr = 0.4
    # update_lr = 1e-3
    update_step = 4
    # 3: 4 updates, 4 accs: [0.53652763 0.57182503 0.60211372 0.62670326]
    update_step_test = 10
    
device = torch.device('cuda')
maml = META(args).to(device)

tmp = filter(lambda x: x.requires_grad, maml.parameters())
num = sum(map(lambda x: np.prod(x.shape), tmp))
print(maml)

print('Total trainable tensors:', num)

from TraceNshot import TraceNshot

#train_idx = np.array([1,2,3,4,5,6,7,8,9,10,11,12]) # swaptions, bodytrack, canneal, vips, facesim, dedup, fluidanimate, freqmine
#test_idx = np.array([1,2,3,4,5,6,7,8,9,10,11,12]) # raytrace, streamcluster
#(num of traces, 199998, 48)
db_train = TraceNshot(data_train=X_train_all, # X_train[train_idx],
                    data_test=X_test_all,
                    label_train=y_train_all, # y_train[train_idx],
                    label_test=y_test_all,
                    batchsz=args.task_num,
                    n_way=args.n_way,
                    k_shot=args.k_shot,
                    k_query=args.k_qry,
                    num_instance=int(sys.argv[1])#203999
                   )

def Load_test(X_test, y_test):
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    test_set = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    return test_loader

def model_eval(maml_net, X_test, y_test):
    correct_total = 0.0
    maml_net.eval()
    with torch.no_grad():
        X_test = torch.from_numpy(X_test).type(torch.cuda.LongTensor)
        y_test = torch.from_numpy(y_test).type(torch.cuda.FloatTensor)
        y_pred = maml_net(X_test)
        y_pred = torch.sigmoid(y_pred)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        correct = torch.eq(y_pred, y_test).all(axis=1).sum().item()
        # correct = torch.eq(y_pred, y_test).sum().item()/16
        real_accuracy = correct/y_test.shape[0] #/199998
    return real_accuracy

acc_list = []      
for step in range(1, args.epoch):

        x_spt, y_spt, x_qry, y_qry = db_train.next() 
        #(32, 256, 48),(32, 256, 48),(32, 256, 16),(32, 256, 16)
        #(task_num, n_way*k_shot, 48),one batch
        #spt for train, qry for evaluation
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
        x_spt, y_spt, x_qry, y_qry = x_spt.clone().type(torch.cuda.LongTensor), y_spt.clone().type(torch.cuda.LongTensor), x_qry.clone().type(torch.cuda.LongTensor), y_qry.clone().type(torch.cuda.LongTensor)
        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
    #    print("ddddd")
        accs = maml(x_spt, y_spt, x_qry, y_qry)
        #print("1accs",accs)
        acc_list.append(accs) 
     #   print("eeeee")
        
        if step % 10 == 0:
            print('step:', step, '\ttraining acc:', accs)

        if step % 40== 0:
            accs = []
            for _ in range(20//args.task_num):
          #  for _ in range(81):
                
                # print("testing accuracy step", _, "started")

                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                x_spt, y_spt, x_qry, y_qry = x_spt.type(torch.cuda.LongTensor), y_spt.type(torch.cuda.LongTensor), \
                                             x_qry.type(torch.cuda.LongTensor), y_qry.type(torch.cuda.LongTensor)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry): 
                    # [256, 48] in [32, 256, 48] 


                    test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)
            torch.save(maml, "./model/maml.md")
            torch.save(maml.net, "./model/maml_net.md")
        # print("step", step, "ended")
#np.save('MAML_type2.npy', acc_list)

#Test
maml_net=torch.load("maml_net.md")
for i in range(13):
    print(model_eval(maml_net, X_test_all[i], y_test_all[i]))

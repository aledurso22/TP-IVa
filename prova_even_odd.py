#%%
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#check if it's the right gpu
print(torch.cuda.get_device_name(device))
print(device)



class one_hidden(nn.Module):
    def __init__(self,d, H):
        super().__init__()
        
        self.layers = nn.Sequential(
      nn.Linear(d, H, bias = False),
      nn.ReLU(),
      nn.Linear(H, 1, bias = False)
    )
        self.d = d
        self.H = H
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x / self.d**0.5)
        return x.reshape((len(x),)) / self.H

batch_size_train = 64
batch_size_test = 1024
tr_split_len = 1024
te_split_len = 1024


random_seed = 1

torch.manual_seed(random_seed)


tr =   torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]))

te =   torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])) 


#here I lower the size
part_tr = torch.utils.data.random_split(tr, [tr_split_len, len(tr)-tr_split_len])[0]
part_te = torch.utils.data.random_split(te, [te_split_len, len(te)-te_split_len])[0]                            

train_loader = torch.utils.data.DataLoader(part_tr,  batch_size = batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(part_te,  batch_size=batch_size_test, shuffle=True)


#I do this just to take the size d
first_batch_train = enumerate(train_loader)
batch_idx, (X_train, y_train) = next(first_batch_train)
d = X_train.shape[-1]
print(X_train.shape)



first_batch_test = enumerate(test_loader)
batch_idx_test, (X_test, y_test) = next(first_batch_test)
print(X_test.shape)
X_test = X_test.to(device)
y_test = y_test.to(device)

L_th = 1e-6
dL_th = 1e-5
maxsteps = 10000000
ps = np.array([900, 950, 1024, 1048, 1072, 1156])
L_trains = []
L_tests = []

#here add the p and print the different
for p in ps:
  student = one_hidden(d*d, p)
  student.to(device)
  optimizer = torch.optim.SGD(student.parameters(), lr=1)
  #my threasholds:

  #I inizialize some variables:
  delta_L = 1
  L_train = 1
  i = 0


  #ho 60000 immagini in totale
  while (L_train>L_th or delta_L > dL_th) and i<maxsteps:
    for batch_idx, (X_train, y_train) in enumerate(train_loader): #per ogni cilo ho 64 (la dim del train_batch) immagini tranne l'ultimo
    #praticamente sto facendo un for per ogni batch
      X_train = X_train.to(device)
      y_train = y_train.to(device)
    

      y_train_label = y_train%2 #1 if odd and 0 if even



      y_pred_train = student(X_train)
      L_train =(1/batch_size_train) * ( (y_train_label - y_pred_train ) ** 2).sum()  
      optimizer.zero_grad()
      L_train.backward()
      optimizer.step() 
      y_pred_train = student(X_train)
      L_train_new = (1/batch_size_train) * ( (y_train_label - y_pred_train ) ** 2).sum()
    #out of the for
    delta_L = abs(L_train - L_train_new)
    if i % 100 == 0: print(f'step {i} - L_train={L_train.item()}')
    i += 1


  counter = 0

  with torch.no_grad():
    #for X_test, y_test in enumerate(test_loader): #ora ho batch da 1000 immagini e quindi in totale avrò 60 cicli
    y_test_label = y_test%2 #1 if odd and 0 if even
    y_pred_test = student(X_test)
    L_test =(1/batch_size_test) * ( (y_test_label - y_pred_test ) ** 2).sum()
    print(f'L_test={L_test.item()}')
  #if device == "cuda:0":
  L_train = L_train.to("cpu")
  L_test = L_test.to("cpu")
  L_trains.append(L_train.detach())
  L_tests.append(L_test.detach())



plt.plot((ps/tr_split_len), np.array(L_trains),'-ok')

plt.xlabel('p/n_train')
plt.ylabel('L train')
plt.yscale('log')
plt.xscale('log')
plt.show()

plt.plot((ps/tr_split_len), np.array(L_tests),'-ok')

plt.xlabel('p/n_train')
plt.ylabel('L test')
plt.yscale('log')
plt.xscale('log')
plt.show()


        





# %%

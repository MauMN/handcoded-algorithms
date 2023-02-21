import torch
import torch.nn as nn

Xs = torch.Tensor([[-1., -1.],
               [-1., 1.],
               [1., -1.],
               [1., 1.]])

y = torch.Tensor([-1., 1., 1., -1.]).reshape(Xs.shape[0], 1)



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(2, 2)
        self.Tanh = nn.Tanh()
        self.linear2 = nn.Linear(2, 1)
        
    def forward(self, input):
        x = self.linear(input)
        Tanh = self.Tanh(x)
        yd = self.linear2(Tanh)
        return yd


xor_network = NeuralNetwork()
epochs = 2000 
mseloss = nn.MSELoss() 
optimizer = torch.optim.Adam(xor_network.parameters(), lr = 0.001) 
all_losses = [] 
current_loss = 0 
plot_every = 50 
 
for epoch in range(epochs): 
   
    # input training example and return the prediction   
    yhat = xor_network.forward(Xs)
    
    # calculate MSE loss   
    loss = mseloss(yhat, y)
      
    # backpropogate through the loss gradiants   
    loss.backward()
    
    # update model weights   
    optimizer.step()
    
    # remove current gradients for next iteration   
    optimizer.zero_grad() 
   
    # append to loss   
    current_loss += loss  
 
    if epoch % plot_every == 0:       
        all_losses.append(current_loss / plot_every)       
        current_loss = 0 
     
    # print progress   
    if epoch % 500 == 0:     
        print(f'Epoch: {epoch} completed')
        
import matplotlib.pyplot as plt
plt.plot(all_losses)
plt.ylabel('Loss')
plt.show()
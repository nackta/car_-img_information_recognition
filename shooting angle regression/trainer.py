import torch
from torch import nn
import tqdm

class cnn_regression(nn.Module):
    def __init__(self):
        super(cnn_regression, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4),

            nn.Conv2d(64, 128, kernel_size = 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(4),

            nn.Conv2d(128, 256, kernel_size = 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4),

            nn.Conv2d(256, 256, kernel_size = 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4)
        )

        self.linear = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        x = self.layer(x)
        x = x.squeeze() 
        x = self.linear(x)
        return x

    
class Trainer():
    def __init__(self, trainloader, testloader, model, optimizer, criterion, device):

        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device  = device
        self._get_scheduler()

    def _get_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=10, gamma=0.5, verbose=True)
        
    def train(self, epoch = 1):
        self.model.train()
        loss_list = []
        for e in range(epoch):
            running_loss = 0.0
            for i, data in tqdm.tqdm(enumerate(self.trainloader, 0)): 
                inputs, target = data
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                   
                outputs = self.model(inputs)
                loss = self.criterion(outputs, target)  

                self.optimizer.zero_grad() 
                loss.backward() 
                self.optimizer.step()

                running_loss += loss.item()
    
            self.scheduler.step()
            running_loss = running_loss / len(self.trainloader)
            loss_list.append(running_loss)
            print('epoch: %d  loss: %.5f' % (e + 1, running_loss))
            
        return loss_list

    def test(self):
        test_loss = 0.0
        self.model.eval()
        for inputs, target in self.testloader:
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            output = self.model(inputs)
            loss = self.criterion(output, target)
            test_loss += loss.item()

        test_loss = test_loss / len(self.testloader)
        print('test_loss: %.5f' %(test_loss))
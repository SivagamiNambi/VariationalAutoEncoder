import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import helper
import torchvision
import cv2
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #encoder

        self.fully_connected_1 = nn.Linear(in_features=784, out_features=450)
        self.fully_connected_2_mu = nn.Linear(in_features=450, out_features=2)
        self.fully_connected_2_sig = nn.Linear(in_features=450, out_features=2)

        #decoder
        self.fully_connected_3 = nn.Linear(in_features=2, out_features=450)
        self.fully_connected_4 = nn.Linear(in_features=450, out_features=784)

    def encode(self, x):
        x = F.relu(self.fully_connected_1(x.view(-1, 784)))
        mu = self.fully_connected_2_mu(x)
        log_varaince = self.fully_connected_2_sig(x)
        return mu, log_varaince

    def reparameterize(self, mu, log_varaince):
        std = torch.exp(0.5*log_varaince)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, mu, log_varaince):
        z = self.reparameterize(mu, log_varaince)
        d = F.relu(self.fully_connected_3(z))
        d = F.sigmoid(self.fully_connected_4(d))

        return d

    def forward(self, x):
        mu, log_variance = self.encode(x)
        decoded_image = self.decode(mu, log_variance)
        return decoded_image, mu, log_variance


def loss_function(decoded_img, img, mu, log_variance):
    bce_loss = F.binary_cross_entropy(decoded_img.view(-1, 784), img.view(-1, 784), reduction='sum')
    KL_Divergence = -0.5 * torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp())

    return bce_loss + KL_Divergence


def train(epochs):
    train_loader, test_loader = helper.load_data()
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    if torch.cuda.is_available():
        model = model.cuda()
    model.train()

    for e in range(epochs):
        train_loss = 0.0
        for data in train_loader:
            input, labels = data

            if torch.cuda.is_available():
                input = input.cuda()

            output, mu, log_variance = model(input)

            loss = loss_function(output, input, mu, log_variance)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item() * input.size(0)

        loss = train_loss / len(train_loader)
        print('Epoch: ' + str(e) + '  Loss :' + str(float(loss)))
    torch.save(model.state_dict(), 'model/trained.h5')

def display():
    model = Net()
    model.load_state_dict(torch.load('model/trained.h5'))
    train_loader, test_loader = helper.load_data()

    test_iter = iter(test_loader)
    test_img, labels = test_iter.next()

    for j in range(3):
        print(j)

        output, mu, logvar = model(test_img)

        output = output.view(100, 1, 28, 28)
        output = output.detach().numpy()
        output_img = np.transpose(output[0], (1, 2, 0)) * 255

        org_input = test_img.numpy()
        org_input_img = np.transpose(org_input[0], (1, 2, 0))*255

        for i in range(1, 20):
             temp = np.transpose(org_input[i], (1, 2, 0)) * 255
             org_input_img = np.hstack((org_input_img, temp))
             temp =  np.transpose(output[i], (1, 2, 0))*255
             output_img = np.hstack((output_img, temp))

        if j == 0:
            input_img = np.vstack((org_input_img, output_img))
        else:
            input_img = np.vstack((input_img, output_img))

    cv2.imwrite('variational' + '.png', input_img)

if __name__ == "__main__":
   # train(20)
    display()








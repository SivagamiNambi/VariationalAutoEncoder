import torch
import torchvision
from torchvision import transforms
import cv2

transforms = transforms.Compose([transforms.ToTensor()])
def image_show(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_data():
    train_set = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms, download=True)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms, download=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=2)

    print('Length of trainset:', len(train_set))
    print('Length of testset:', len(test_set))

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = load_data()

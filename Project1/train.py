import torch
import torch.nn as nn
from torchvision import transforms, datasets
import tqdm
import argparse
from model import RobustModel

def test(data_loader, model):
    model.eval()
    n_predict = 0
    n_correct = 0
    with torch.no_grad():
        for X, Y in tqdm.tqdm(data_loader):
            y_hat = model(X)
            y_hat.argmax()
            
            _, predicted = torch.max(y_hat, 1)
            
            n_predict += len(predicted)
            n_correct += (Y == predicted).sum()
            
    accuracy = n_correct/n_predict
    print(f"Accuracy: {accuracy} ()")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train RGB mnist dataset with CNN model')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=20)
    args = parser.parse_args()
    
    # model
    model = RobustModel()
    criterian = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # data
    trans = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor()])
    train_data = datasets.MNIST('C:/Users/minkyeong/Desktop/DeepLearning/Project1/data/train', train=True, download=True, transform=trans)
    train_set, val_set = torch.utils.data.random_split(train_data, [50000, 10000])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size)
    
    
    for epoch in range(args.epoch):
        model.train()
        cost = 0
        n_batches = 0
        
        for X, Y in tqdm.tqdm(train_loader): # 미니 배치 단위로 꺼내온다
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterian(y_hat, Y)
            loss.backward()
            optimizer.step()
            
            cost += loss.item()
            n_batches += 1
        
        cost /= n_batches
        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch+1, cost))
        test(dev_loader, model)
        torch.save(model.state_dict(), './model/checkpoint_'+str(epoch+1)+'epoch.pt' )
        

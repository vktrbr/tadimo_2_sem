import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %%
mnist_train = datasets.FashionMNIST('./fashion_mnist/train', train=True,
                                    transform=transforms.Compose([transforms.ToTensor()]))
# %%
mnist_test = datasets.FashionMNIST('./fashion_mnist/test', train=False,
                                   transform=transforms.Compose([transforms.ToTensor()]))

mnist_train.data = mnist_train.data.float()
# %%
device = 'cpu'  # 'mps' if torch.has_mps else 'cpu'
# %%
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(28 * 28, 2048),
    torch.nn.LeakyReLU(0.05),
    torch.nn.Linear(2048, 32),
    torch.nn.LeakyReLU(2),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(32, len(mnist_train.classes))
).to(device)


def test_accuracy(_model, _test_loader, _loss_func) -> [float, float]:
    """ Возвращает два числа -- точность и лосс """
    acc = 0
    _loss = 0
    with torch.no_grad():
        for j, (x, y) in enumerate(_test_loader, 1):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            acc += torch.sum(torch.eq(pred.argmax(dim=1).long().to('cpu'), y.to('cpu')))
            _loss += _loss_func(pred, y)

    return acc / _test_loader.dataset.data.shape[0], _loss / j


# %%
def train(_model, train_data, test_data, epochs: int = 30, batch_size: int = 20_000) -> [torch.nn.Module, pd.DataFrame]:
    optimizer = torch.optim.Adam(_model.parameters(), weight_decay=0.00001)
    loss = torch.nn.CrossEntropyLoss()

    loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    log = pd.DataFrame(columns=['epoch', 'train_loss', 'test_accuracy', 'test_loss'])

    for i in range(epochs):
        epoch_loss = 0
        model.train()
        for j, (x, y) in enumerate(loader, 1):
            x = x.data.to(device)
            y = y.data.to(device)
            y_pred = _model(x.data)
            running_loss = loss(y_pred, y)
            running_loss.backward()

            epoch_loss += running_loss.item()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        test_acc, test_loss = test_accuracy(model, test_loader, loss)
        log.loc[i] = [i + 1, epoch_loss / j, test_acc.item(), test_loss.item()]

        print(f'EPOCH: {i + 1 :3d}  |  LOSS: {epoch_loss / j: .4f}', end='  |  ')
        print(f'TEST LOSS: {test_loss:0.4f}  |  TEST ACCURACY: {test_acc:0.4f}')

    return _model, log


# %%
model, log = train(model, mnist_train, mnist_test, 10)

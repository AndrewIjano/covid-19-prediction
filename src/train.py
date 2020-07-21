import torch
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


def validate(train_set, test_set, model, optimizer, criterion, plot=False):
    model = train_network(train_set, model, optimizer, criterion)

    (test_set, model, criterion) = transfer_to_device(test_set, model, criterion)
    predict = get_predict(model, criterion)

    X_test, y_test = test_set
    y_test_pred, test_loss = predict(X_test, y_test)
    if plot:
        plot_confusion_matrix(y_test, y_test_pred)
    return calculate_accuracy(y_test, y_test_pred)


def train_network(data, model, optimizer, criterion):
    (data, model, criterion) = transfer_to_device(data, model, criterion)
    X, y = data
    predict = get_predict(model, criterion)

    for epoch in range(1000):
        y_pred, train_loss = predict(X, y)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    return model


def transfer_to_device(data, model, criterion):
    ''' Transfer objects to GPU if available, otherwise transfer to CPU '''

    def get_device():
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    device = get_device()

    def object_to_device(obj):
        return obj.to(device)

    data = tuple(map(object_to_device, data))
    model = model.to(device)
    criterion = criterion.to(device)

    return (data, model, criterion)


def get_predict(model, criterion):
    def predict(X, y):
        y_pred = model(X)
        y_pred = torch.squeeze(y_pred)
        loss = criterion(y_pred, y)
        return y_pred, loss

    return predict


def plot_confusion_matrix(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.ge(.5).view(-1).cpu().detach().numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    df_cm = pd.DataFrame([[tp, fp], [fn, tn]],
                         ['$\hat{y} = $ Reagente',
                             '$\hat{y} =  $ Não Reagente'],
                         ['$y = $ Reagente', '$y = $ Não Reagente'])

    plt.figure(figsize=(6, 6))
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}, cmap='coolwarm')
    plt.show()


def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return ((y_true == predicted).sum().float() / len(y_true)).item()

import statistics as stat
import torch
from .model import Net
from . import dataset, train

N_HL1 = 5
N_HL2 = 3
LEARNING_RATE = 0.001
DATASET_PATH = 'data/einstein.out.csv'
TRAINED_MODEL_PATH = 'data/models/trained_model'


def create_model(X):
    _, features_count = X.size()
    model = Net(features_count, N_HL1, N_HL2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    return model, optimizer, criterion


def print_results(accuracies):
    metrics = {
        'mean: ': stat.mean(accuracies),
        'stdev:': stat.stdev(accuracies),
        'max:  ': max(accuracies),
        'min:  ': min(accuracies)
    }
    for k in metrics.keys():
        print(f'  {k} {metrics[k]:4.3f}')


def main():
    print('loading dataset...')
    data = dataset.load(DATASET_PATH)

    for exam_name, (X, y) in dataset.get_exam_data(data):
        print('dataset size:')
        print(f' X:{X.shape} y:{y.shape}')
        print('cross validating model...')
        accuracies = []
        for idx, (train_set, test_set) in enumerate(dataset.get_cross_validaion_sets(X, y)):
            print(f'  repetition {idx}', end='')

            model, optimizer, criterion = create_model(train_set[0])
            accuracy = train.validate(
                train_set, test_set, model, optimizer, criterion)

            print(f'  acc: {accuracy:4.3f}')
            accuracies += [accuracy]
        print_results(accuracies)

        print('training final model with all data...')
        X, y = dataset.prepare_dataset(X, y)
        model, optimizer, criterion = create_model(X)
        model = train.train_network((X, y), model, optimizer, criterion)

        torch.save(model, f'{exam_name}_{TRAINED_MODEL_PATH}')
        print(f'model trained, saved at {TRAINED_MODEL_PATH}_{exam_name}.pt')


if __name__ == '__main__':
    main()

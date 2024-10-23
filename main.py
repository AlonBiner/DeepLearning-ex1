import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

size_of_peptide = 9
amino_acid_encoding = "ARNDCEQGHILKMFPSTWYV"
amino_acid_to_index_mapping = {amino_acid: index for index, amino_acid in enumerate(amino_acid_encoding)}

class PeptideDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        return sample, label

class MLP(nn.Module):
    def __init__(self, sizes_of_inputs, activation_func):
        super(MLP, self).__init__()
        self.layers = [nn.Linear(sizes_of_inputs[i], sizes_of_inputs[i + 1]) for i in range(len(sizes_of_inputs) - 1)]
        self.layers = nn.ModuleList(self.layers)
        self.activation_func = activation_func

    def forward(self, x):
        for layer in self.layers[:len(self.layers) - 1]:
            x = self.activation_func(layer(x))
        x = self.layers[-1](x)
        return F.softmax(x, dim=1)
        #return F.softmax(x, dim=1)

def preprocess_data(peptides_file_path):
    """
    Preprocesses data
    :param data:
    :return:
    """
    #Get rid of null values
    data = pd.read_csv(peptides_file_path, sep='\t', names=['peptide'])
    data = data.dropna()

    # Remove invaid values
    def is_valid_peptide(peptide):
        return len(peptide) == size_of_peptide and all(char in amino_acid_encoding for char in peptide)

    data = data[data['peptide'].apply(is_valid_peptide)]

    def peptide_to_index(peptide):
        return [amino_acid_to_index_mapping[char] for char in peptide]

    # Encode each peptide using one hot encode
    data = data['peptide'].apply(peptide_to_index)
    data = torch.tensor(data.tolist(), dtype=torch.long)
    data = F.one_hot(data, num_classes=len(amino_acid_encoding)).float()
    data = data.view(data.size(0), -1)
    data = data.to(torch.float32)

    return data

def load_data():
    """
    loads peptides
    :return:
    """
    # Preprocess positive peptides
    positive_peptides_file_path="ex1 data/pos_A0201.txt"
    positive_peptides = preprocess_data(positive_peptides_file_path)

    # Preprocess negative peptides
    negative_peptides_file_path = "ex1 data/neg_A0201.txt"
    negative_peptides = preprocess_data(negative_peptides_file_path)

    # Merge peptides into a single traning data set
    #data = pd.concat([positive_peptides, negative_peptides], ignore_index=True)

    return positive_peptides, negative_peptides

def create_dataset(positive_samples, negative_samples):
    # Unify positive and negative samples
    samples = torch.cat([positive_samples, negative_samples])

    # Unify positive and negative labels
    positive_labels = torch.tensor([[0, 1]] * positive_samples.shape[0], dtype=torch.float32)
    negative_labels = torch.tensor([[1, 0]] * negative_samples.shape[0], dtype=torch.float32)

    labels = torch.cat([positive_labels, negative_labels])

    # Create peptide dataset
    dataset = PeptideDataset(samples, labels)

    return dataset


def split_data(positive_samples, negative_samples):
    """
    Split the data into training and test sets using PyTorch.
    """
    # Create dataset of peptides
    dataset = create_dataset(positive_samples, negative_samples)

    # Split dataset randomly such that 10% is in test set and the rest is train set
    split_ratio = 0.1
    test_size = int(len(dataset) * split_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def calculate_class_weights(labels):
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return class_weights

def calculate_weights(train_set, test_set):
    # Calculate class weights for train set
    train_labels = torch.cat([train_set.dataset.labels[i] for i in train_set.indices]).long()
    train_class_weights = calculate_class_weights(train_labels)
    train_class_weights = train_class_weights.float()

    # Calculate class weights for test set
    test_labels = torch.cat([test_set.dataset.labels[i] for i in test_set.indices]).long()
    test_class_weights = calculate_class_weights(test_labels)
    test_class_weights = test_class_weights.float()

    return train_class_weights, test_class_weights

def create_mlp():
    # Create MLP
    size_of_input = size_of_peptide * len(amino_acid_encoding)
    layer_dimensions = [size_of_input, 100, 50, 2]

    activation_functions = [lambda x: x, torch.relu]
    model = MLP(layer_dimensions, activation_functions[1])
    return model


def train_mlp(mlp, train_loader, train_criterion, optimizer):
    total_loss = 0.0
    mlp.train(True)
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = mlp(inputs)
        train_loss = train_criterion(outputs, targets)
        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.item()
    average_loss = total_loss / len(train_loader.dataset)
    return average_loss


def create_train_losses(num_epochs, model, train_loader, train_criterion, optimizer):
    train_losses = []
    for epoch in range(num_epochs):
        train_loss = train_mlp(model, train_loader, train_criterion, optimizer)
        train_losses.append(train_loss)
    return train_losses


def test_mlp(mlp, test_loader, test_criterion):
    total_loss = 0.0
    mlp.eval()
    #correct = 0
    #total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = mlp(inputs)
            loss = test_criterion(outputs, targets)
            total_loss += loss.item()
            #_, predicted = torch.max(outputs, 1)
            #total += targets.size(0)
            #correct += (predicted == targets).sum().item()
    average_loss = total_loss / len(test_loader.dataset)
    #accuracy = correct / total
    return average_loss#, accuracy

    # total_loss = 0.0
    # mlp.train(False)
    # with torch.no_grad():
    #     for inputs, targets in test_loader:
    #         outputs = mlp(inputs)
    #         total_loss = test_criterion(outputs, targets)
    # average_loss = total_loss / len(test_loader.dataset)
    # return average_loss


def create_test_losses(num_epochs, model, test_loader, test_criterion):
    test_losses = []
    for epoch in range(num_epochs):
        test_loss = train_mlp(model, test_loader, test_criterion)
        test_losses.append(test_loss)
    return test_losses

def preprocess_spike():
    with open("ex1 data/spike.txt", 'r') as file:
        data = file.read().replace('\n', '').replace(' ', '')

    spike_peptides = [data[i:i+size_of_peptide] for i in range(len(data) - size_of_peptide + 1)]
    # spike_peptides = [spike_peptides[i:i + 9] for i in range(len(spike_peptides) - 8)]

    def peptide_to_index(peptide):
        return [amino_acid_to_index_mapping[char] for char in peptide]
    # problem is here...
    #print(spike_peptides)
    spike_peptides_tensor = [peptide_to_index(peptide) for peptide in spike_peptides]
    spike_peptides_tensor = torch.tensor(spike_peptides_tensor, dtype=torch.long)
    spike_peptides_tensor = F.one_hot(spike_peptides_tensor, num_classes=len(amino_acid_encoding)).float()
    spike_peptides_tensor = spike_peptides_tensor.view(spike_peptides_tensor.size(0), -1)
    return spike_peptides, spike_peptides_tensor

def load_spike():
    spike_peptides, spike_peptides_tensor = preprocess_spike()
    return spike_peptides, spike_peptides_tensor

    #spike_peptides = preprocess_spike(spike_peptides_file_path)

def predict_3_top_acids(model):
    spike_peptides, spike_peptides_tensor = load_spike()
    model.eval()
    with torch.no_grad():
        predictions = model(spike_peptides_tensor)
    predictions = predictions[:, 1]

    top_indices = torch.topk(predictions, 3).indices
    top_peptides = [spike_peptides[i] for i in top_indices]
    top_predictions = [predictions[i].item() for i in top_indices]
    print("Top 3 most detectable peptides:")
    for i, (peptide, score) in enumerate(zip(top_peptides, top_predictions)):
        print(f"{peptide} with detection probability {top_predictions[i]}")


def main():
    # Path to the data file
    # Load peptides
    positive_peptides, negative_peptides = load_data()

    # Split the data into training and test sets
    train_set, test_set = split_data(positive_peptides, negative_peptides)

    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Calculate class weights
    train_class_weights, test_class_weights = calculate_weights(train_set, test_set)

    # Create MLP
    model = create_mlp()

    train_criterion = nn.CrossEntropyLoss(weight=train_class_weights)
    test_criterion = nn.CrossEntropyLoss(weight=test_class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #train_losses = create_train_losses(num_epochs, model, train_loader, train_criterion, optimizer)
    #test_losses = create_test_losses(num_epochs, model, test_loader, test_criterion)

    num_epochs = 100
    train_losses = []
    test_losses = []
    #test_accuracies = []

    for epoch in range(num_epochs):
        train_loss = train_mlp(model, train_loader, train_criterion, optimizer)
        #test_loss, test_accuracy = test_mlp(model, test_loader, test_criterion)
        test_loss = test_mlp(model, test_loader, test_criterion)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        #test_accuracies.append(test_accuracy)
        # print(
        #     f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss},")

        # print(
        #     f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss},"
        #     f" Test Accuracy: {test_accuracy * 100}%")

    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Testing Loss over Epochs')
    plt.show()

    predict_3_top_acids(model)


if __name__ == "__main__":
    main()


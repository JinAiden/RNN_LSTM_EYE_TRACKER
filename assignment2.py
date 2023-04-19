import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

global_figsize = (10, 5)


def get_one_data_point(csv_filename):
    """
    Returns two dataframe object containing the data for the specified participant and trial, and the questionnaire data.
    """

    dirname = "./dataset/"
    csv_filename = dirname + csv_filename
    if os.path.isfile(csv_filename) and "csv" in csv_filename:
        df_original_dataset = pd.read_csv(csv_filename)

        if "II" in csv_filename:
            df_original_questionnaire = pd.read_csv(
                dirname + "Questionnaire_datasetIA.csv")
        elif "III" in csv_filename:
            df_original_questionnaire = pd.read_csv(
                dirname + "Questionnaire_datasetIB.csv")

        return df_original_dataset, df_original_questionnaire
    return None, None


def get_filename_list():
    """
    Returns a list of all the filenames in the current directory.
    """
    filenames = []
    # go into the ./dataset dir and get all the filenames contianing "csv"
    for file in os.listdir("./dataset"):
        if file.endswith(".csv"):
            filenames.append(file)
    return sorted(filenames)


def preprocessing(df_original: pd.DataFrame, scale=False, verbose=False):

    df_preprocessed_dataset = df_original.copy()
    # remove some columns
    # In my mind, there are some columns that apparently would not be useful for modelling.
    columns_to_be_dropped = ["Export date", "Recording name", "Recording date", "Recording date UTC", "Recording start time", "Recording start time", "Recording start time UTC",
                             "Recording software version", "Project name", "Participant name", "Recording Fixation filter name", "Timeline name", "Recording monitor latency", "Eye movement type"]
    # columns_to_be_dropped += ["Eyetracker timestamp", "Computer timestamp", "Recording timestamp"]

    # the media information does not matter
    columns_to_be_dropped += ['Presented Stimulus name',
                              'Presented Media name', 'Presented Media width', 'Presented Media height']

    # columns with std = 0
    columns_with_zero_std = [
        "Recording duration", "Recording resolution height", "Recording resolution width"]

    columns_to_be_dropped += columns_with_zero_std

    if verbose:
        print("There would be", len(columns_to_be_dropped),
              "columns to be dropped.")
        print(df_preprocessed_dataset.shape)

    try:
        df_preprocessed_dataset = df_preprocessed_dataset.drop(
            columns_to_be_dropped, axis=1)
        # also drop the first unnamed column
        df_preprocessed_dataset.pop(df_preprocessed_dataset.columns[0])
    except KeyError:
        pass

    # replace all the , to . in the number values
    df_preprocessed_dataset = df_preprocessed_dataset.replace(
        to_replace=r',', value='.', regex=True)

    # remove lines where eye tracker is detected as invalid
    df_preprocessed_dataset = df_preprocessed_dataset[(df_preprocessed_dataset["Validity left"] == "Valid") & (
        df_preprocessed_dataset["Validity right"] == "Valid")]
    df_preprocessed_dataset.drop(
        columns=['Validity left', 'Validity right'], inplace=True)

    if verbose:
        print(df_preprocessed_dataset.shape)
        print(df_preprocessed_dataset.columns)

    # one-hot encode some features
    try:
        one_hot1 = pd.get_dummies(df_preprocessed_dataset['Event'])
        one_hot2 = pd.get_dummies(df_preprocessed_dataset['Event value'])
        one_hot3 = pd.get_dummies(df_preprocessed_dataset['Sensor'])

        df_preprocessed_dataset.drop(
            columns=['Event value', 'Event', 'Sensor'], inplace=True)
        df_preprocessed_dataset = pd.concat(
            [df_preprocessed_dataset, one_hot1, one_hot2, one_hot3], axis=1)

    except KeyError:
        print("KeyError while one-hot encoding.")

    # handle NA
    # Fill all the NaN values using bfill method with limit equals to 1
    df_preprocessed_dataset.fillna(method='bfill', limit=1, inplace=True)

    if verbose:
        print(df_preprocessed_dataset.shape)

    # scale the dataset
    if scale:
        standard_scaler = StandardScaler()
        df_preprocessed_dataset = standard_scaler.fit_transform(
            df_preprocessed_dataset)

    # return df_preprocessed_dataset
    return df_preprocessed_dataset["Gaze point X"]


def plot_data_point(preprocessed_dataset, y="Computer timestamp"):
    plt.plot(preprocessed_dataset, label=y)
    plt.title(y + " vs. Gaze point X")
    plt.xlabel("Gaze point X")
    plt.ylabel(y)
    plt.legend()
    plt.rcParams["figure.figsize"] = (3, 3)
    plt.show()


data_set_filenames = get_filename_list()

trian_ratio = 0.8
dev_train_ratio = 0.1
test_train_ratio = 0.1

train_N = int(len(data_set_filenames) * trian_ratio)
dev_N = int(len(data_set_filenames) * dev_train_ratio)
test_N = int(len(data_set_filenames) * test_train_ratio)

train_set = data_set_filenames[0: train_N]
dev_set = data_set_filenames[train_N: train_N + dev_N]
test_set = data_set_filenames[train_N + dev_N: train_N + dev_N + test_N]

df_original_dataset0, df_ground_truth0 = get_one_data_point(train_set[0])
df_original_dataset2, df_ground_truth2 = get_one_data_point(train_set[2])


df_preprocessed_datasetX = preprocessing(df_original_dataset0)
# plot_dataset(df_preprocessed_dataset)

# df_preprocessed_datasetX = df_preprocessed_dataset["Gaze point X"]

df_preprocessed_datasetX.iloc[0:10]


def padding_data(data):
    X = np.array(data)
    X.reshape(-1, 1)
    X = np.concatenate(
        (np.array(range(len(X))).reshape(-1, 1), X.reshape(-1, 1)), axis=1)
    return X

# train_X = padding_data(df_preprocessed_datasetX)


plt.rcParams["figure.figsize"] = (12, 5)
plt.plot(df_preprocessed_datasetX, label="Gaze point X")
# set the label to be upper left
plt.rcParams['legend.loc'] = 'upper left'
plt.legend()
plt.show()


plt.rcParams["figure.figsize"] = global_figsize
df_preprocessed_dataset2 = preprocessing(df_original_dataset2)
plot_data_point(df_preprocessed_dataset2)


def get_participant_data(participant_i, dirname="./dataset/"):

    data_names = get_filename_list()
    participant_data_names = [
        data for data in data_names if f"participant_{participant_i}_" in data]
    participant_data = []
    ground_truth = None

    if len(participant_data_names) == 0:
        print("No data for participant", participant_i)
        return [], None

    _, ground_truth = get_one_data_point(participant_data_names[0])

    for data_name in participant_data_names:
        with open(dirname + data_name, 'r') as f:

            df_original_dataset = pd.read_csv(f)
            gaze_point_x = preprocessing(df_original_dataset)

            # gaze_point_x = df_preprocessed_dataset.fillna(method="ffill")
            # computer_timestamp = df_preprocessed_dataset["Computer timestamp"]

            # train_X = np.concatenate((computer_timestamp.values.reshape(-1, 1), gaze_point_x.values.reshape(-1, 1)), axis=1)
            # adding index
            train_X = padding_data(gaze_point_x)
            participant_data.append(train_X)

    return participant_data, ground_truth

# create dataset with look back = 1

def create_dataset(i, look_back=1):

    participant_data, ground_truth = get_participant_data(i)
    if participant_data == []:
        raise TypeError

    ground_truth = pd.read_csv(
        "./dataset/Questionnaire_datasetIA.csv", encoding="ISO-8859-1").iloc[i, 46]

    N = len(participant_data[0])
    dataX, dataY = [], []

    for i in range(N - look_back - 1):

        x = participant_data[0][i:(i + look_back), 0]
        dataX.append(x.item())

        dataY.append(participant_data[0][i + look_back, 1].item())

    if len(dataX) < 4000:
        data_x_np = np.pad(np.array(dataX), (0, 4000 - len(dataX)), 'constant')
        data_y_np = np.pad(dataY, 0, (0, 4000 - len(dataY)),
                           'constant', constant_values=(0, 0))
    else:
        data_x_np = np.array(dataX[:4000], dtype=np.int64)
        data_y_np = np.array(dataY[:4000], dtype=np.float64)

    concatentated_data = np.concatenate(
        (data_x_np.reshape(4000, 1), data_y_np.reshape(4000, 1)), axis=1)

    return torch.Tensor(concatentated_data), torch.Tensor([ground_truth])


# an example of visualising one data point
data, gt = create_dataset(2)
plt.rcParams["figure.figsize"] = (12, 5)
plt.plot(data[:, 0], data[:, 1])

# x label
plt.xlabel("Computer timestamp")
plt.ylabel("Gaze Point X")

plt.title(f"label={gt}")
plt.show()



torch.random.manual_seed(123)

# train
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2000)
        self.fc2 = nn.Linear(2000, 1)
        self.dropout = nn.Dropout(dropout)

        self.criterion = nn.MSELoss()
        self.train_range = range(1, 51)
        self.eval_range = range(51, 60)
        self.iteration_loss = []

    def forward(self, data):
        embedded = self.dropout(self.embedding(data))
        embedded = embedded.unsqueeze(1)
        lstm_output, (hidden, cell) = self.lstm(embedded)
        output = self.fc2(self.fc(self.dropout(lstm_output[:, -1, :])))
        return output
    
    def start_train(self, lr=5e-4, n_epochs=2):

        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in tqdm(range(n_epochs), desc="Epoch"):
            epoch_loss = 0
            self.train()
            for i in tqdm(self.train_range, desc="Train"):

                try:
                    data, labels = create_dataset(i)
                except TypeError:
                    continue

                data = data.to(torch.int)

                optimizer.zero_grad()

                # Forward pass
                predictions = self.forward(data[:, 1])

                loss = self.criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    print(f"Itteration {i}: RMSE Loss {torch.sqrt(loss).item()}")
                    self.iteration_loss.append(torch.sqrt(loss).item())
                    epoch_loss += torch.sqrt(loss).item()

            print(f'Epoch {epoch + 1}: Loss {epoch_loss / (epoch + 1)}')
            
            # update the learning rate according to the epoch
            if (epoch + 1) / n_epochs == 0.5:

                # update the learning rate of optimizer
                lr /= 5

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                print(f"Learning rate is updated to {lr}")
    
    def evaluate(self, i):

        try:
            data, label = create_dataset(i)
        except TypeError:
            return -1, -1, -1

        self.eval()
        with torch.no_grad():
            data = data.to(torch.int)
            predictions = self.forward(data[:, 1])
            return predictions, label, torch.sqrt(self.criterion(predictions, label))

lstm_model = LSTMModel(4000, 4000, 4000, 1, 2, 0.05)
lstm_model.start_train()



## Plot
# loss for iterations
plt.plot(lstm_model.iteration_loss, label="RMSE Loss")
plt.title("RMSE Loss after each iteration")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("RMSE Loss")
plt.rcParams["legend.loc"] = 'upper right'
plt.show()

# validate on test set

plt.ylim(0, 160)
for j in range(50, 58):
    pred, label, _ = lstm_model.evaluate(j)

    if isinstance(pred, int) and pred == -1:
        continue

    # plot the prediction vs. the label
    plt.plot(j, torch.mean(pred).item(), marker="o", color="red")
    plt.plot(j, label, marker="o", color="blue")


plt.legend()
plt.show()


# Linear Model on the Questionnaire 

def get_questionnaire_dataset(dirname="./dataset/", ):

    # get the questionnaire A
    filename_a = "Questionnaire_datasetIA.csv"
    with open(dirname + filename_a, 'r', encoding="ISO-8859-1") as f:
        df_questionnaire_datasetA = pd.read_csv(f)

    # get the questionnaire B
    filename_b = "Questionnaire_datasetIB.csv"
    with open(dirname + filename_b, 'r', encoding="ISO-8859-1") as f:
        df_questionnaire_datasetB = pd.read_csv(f)
    
    # combine the two questionnaire datasets
    df_combined_questionnaire_dataset = pd.concat([df_questionnaire_datasetA, df_questionnaire_datasetB], axis=0)

    return df_combined_questionnaire_dataset

def extract_X_y(df_dataset):
    return df_dataset.iloc[:, 6:46].to_numpy(), df_dataset.iloc[:, 46].to_numpy()

def get_train_test_data_XY(train=False, test=False):

    if train or test:
        data_set = get_questionnaire_dataset()
        data_set.fillna(data_set.mean(), inplace=True)
        # split the dataset into train and test, with train : test = 8 : 2
        train_set, test_set = train_test_split(data_set, test_size=0.3, random_state=42)

        if train and not test:
            trainX, trainY = extract_X_y(train_set)
            return trainX, trainY

        if test and not train:
            testX, testY = extract_X_y(test_set)
            return testX, testY
        
        if train and test:
            trainX, trainY = extract_X_y(train_set)
            testX, testY = extract_X_y(test_set)
            return (trainX, trainY, testX, testY)

        # assert the trainX and testX have the same length
        # assert (trainX.shape[1] == testX.shape[1])
    else:
        return None

trainX, trainY, testX, testY = get_train_test_data_XY(train=True, test=True)


from sklearn.metrics import mean_squared_error


def save_metric_to_global(model, rmse, r2):

    # Since each model is only evaluated only once on the test set,
    # the model name is used as a key for the set
    model_name = model.__class__.__name__
    if model_name not in global_metric["models"]:
        global_metric["models"].add(model_name)
        global_metric['RMSE'].append((model_name, rmse))
        global_metric['R2'].append((model_name, r2))


def metric(model, X, Y):

    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y_pred, Y))
    r2_score = model.score(X, Y)

    save_metric_to_global(model, rmse, r2=r2_score)

    print("The model's RMSE: {:.4f}".format(rmse))
    print("The model's R^2 score: {:.4f}".format(r2_score))

    # add scores into global metric


def plot_pred_vs_label(model, X, y, train=True):

    # Customize plot style
    # the style of the reference page: https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html

    # the meaning of the code could be simply inferred by its name or attribute
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['axes.edgecolor'] = 'gray'
    plt.rcParams['axes.facecolor'] = '#F5F5F5'
    plt.rcParams['axes.labelcolor'] = 'gray'

    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.solid_capstyle'] = 'round'

    plt.rcParams['xtick.color'] = 'gray'
    plt.rcParams['ytick.color'] = 'gray'

    plt.rcParams['text.color'] = 'gray'
    plt.rcParams['font.family'] = 'serif'

    dataset = ""
    if train:
        dataset = "Training Set"
    else:
        dataset = "Test Set"

    title = f'{model.__class__.__name__} Model Performance - Actual vs Predicted on {dataset}'
    print(f'Figure:\n{title}.')
 

    # the figure size set up
    plt.figure(figsize=(10, 5))

    # plot for the label
    plt.plot(range(0, len(X)), y, color='green', label='Actual', alpha=0.8)

    y_pred = model.predict(X)
    plt.plot(range(0, len(X)), y_pred, color='orange', label='Predicted', alpha=0.8)

    plt.legend(frameon=True, edgecolor='gray', facecolor='#F5F5F5', framealpha=1, loc='lower left')

    plt.title(title)
    plt.ylabel('The Empathy Score')
    plt.xlabel('The Participant Index')

    plt.show()

# accept the model's class and its arguments
# and evaluate the training performance with plot and metric
# return the trained model
def pipeline(args, plot=True):

    # no data is provided
    model, X, Y = None, None, None

    train = False

    # for training
    if (isinstance(args, dict)) and ("class" in args):

        train = True

        model_class = args["class"]
        del args["class"]
        X, Y = get_train_test_data_XY(train=True)

        # init the model
        model = model_class(**args)
        model.fit(X, Y)
    # for evaluating
    else:
        # get the model from the args
        model = args
        X, Y = get_train_test_data_XY(test=True)


    metric(model, X, Y) # print metric information
    if plot:
        plot_pred_vs_label(model, X, Y, train)

    return model


# Lineare Regression Model

from sklearn.linear_model import LinearRegression

# setting up args while initializing the linear regression model
lr_args = {
    "class": LinearRegression,
    "fit_intercept": True,
    "normalize": True, # 
    "n_jobs": 4,
}

lr_trained = pipeline(lr_args)
# when the incomming arugment is a single model instance,
# the pipeline automatically recognize it as a trained model
pipeline(lr_trained)

## Ridge Regression Model
from sklearn.linear_model import Ridge

rd_args = {
    "class": Ridge,
    "alpha": 1.05,
    # "normalize": True
}

rd_trained = pipeline(rd_args)
pipeline(rd_trained)

## Random Forest Model
from sklearn.ensemble import RandomForestRegressor

rf_args = {
    "class": RandomForestRegressor,
    "n_estimators": 1000,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": 10,
}

rf_trained = pipeline(rf_args)

pipeline(rf_trained)

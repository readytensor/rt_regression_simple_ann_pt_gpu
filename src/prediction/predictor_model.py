import os
import warnings
from typing import Callable, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from logger import get_logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

MODEL_PARAMS_FNAME = "model_params.save"
MODEL_WTS_FNAME = "model_wts.save"
HISTORY_FNAME = "history.json"

logger = get_logger(task_name="pt_model_training")

device = "cuda:0" if T.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


def get_activation(activation: str) -> Callable:
    """
    Return the activation function based on the input string.

    This function returns a callable activation function from the
    torch.nn.functional package.

    Args:
        activation (str): Name of the activation function.

    Returns:
        Callable: The requested activation function. If 'none' is specified,
        it will return an identity function.

    Raises:
        Exception: If the activation string does not match any known
        activation functions ('relu', 'tanh', or 'none').

    """
    if activation == "tanh":
        return F.tanh
    elif activation == "relu":
        return F.relu
    elif activation == "none":
        return lambda x: x  # Identity function, doesn't change input
    else:
        raise ValueError(
            f"Error: Unrecognized activation type: {activation}. "
            "Must be one of ['relu', 'tanh', 'none']."
        )


def get_optimizer(
    optimizer_name: str, parameters, lr: float = 0.001
) -> optim.Optimizer:
    """
    Return the optimizer based on the input string.

    This function returns a PyTorch optimizer initialized with the given parameters
    and learning rate.

    Args:
        optimizer_name (str): Name of the optimizer.
        parameters (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate for the optimizer.

    Returns:
        optim.Optimizer: The requested optimizer.

    Raises:
        Exception: If the optimizer_name string does not match any known
        optimizers ('sgd', 'adam', 'rmsprop').

    """
    if optimizer_name == "sgd":
        return optim.SGD(parameters, lr=lr)
    elif optimizer_name == "adam":
        return optim.Adam(parameters, lr=lr)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(parameters, lr=lr)
    else:
        raise ValueError(
            f"Error: Unrecognized optimizer type: {optimizer_name}. "
            "Must be one of ['sgd', 'adam', 'rmsprop']."
        )


class Net(T.nn.Module):
    def __init__(self, D: int, activation: str) -> None:
        """
        Initialize the neural network.

        Args:
            D (int): Dimension of input data.
            activation (str): Activation function to be used in hidden layers.

        Returns:
            None
        """
        super(Net, self).__init__()
        M1 = max(100, int(D * 4))
        M2 = max(30, int(D * 0.5))
        self.activation = get_activation(activation)
        self.hid1 = T.nn.Linear(D, M1)
        self.hid2 = T.nn.Linear(M1, M2)
        self.oupt = T.nn.Linear(M2, 1)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """
        Forward pass through the network.

        Args:
            x (T.Tensor): Input to the network.

        Returns:
            x (T.Tensor): Output of the network.
        """
        x = self.activation(self.hid1(x))
        x = self.activation(self.hid2(x))
        x = self.oupt(x)
        return x

    def get_num_parameters(self) -> int:
        """
        Calculate the total number of parameters in the network.

        Returns:
            pp (int): Total number of parameters in the network.
        """
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp


class CustomDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the dataset.

        Args:
            x (np.ndarray): Input data.
            y (np.ndarray): Corresponding labels.

        Returns:
            None
        """
        self.x = np.array(x)
        self.y = np.array(y)

    def __getitem__(self, index: int) -> tuple:
        """
        Get one item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: A tuple containing the item and its corresponding label.
        """
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.x)


def get_loss(
    model: T.nn.Module,
    device: str,
    data_loader: T.utils.data.DataLoader,
    loss_function: T.nn.modules.loss._Loss,
) -> float:
    """
    Calculate the average loss over the dataset.

    Args:
        model (T.nn.Module): The model to calculate the loss on.
        device (str): The device on which the calculations will be performed.
        data_loader (T.utils.data.DataLoader): The data loader providing the data.
        loss_function (T.nn.modules.loss._Loss): The loss function to use.

    Returns:
        float: The average loss.
    """
    model.eval()
    loss_total = 0
    with T.no_grad():
        for data in data_loader:
            inputs = data[0].to(device).float()
            targets = data[1].to(device).float()
            output = model(inputs.view(inputs.shape[0], -1))
            loss = loss_function(output, targets)
            loss_total += loss.item()
    return loss_total / len(data_loader)


class Regressor:
    """A wrapper class for the ANN Regressor in PyTorch."""

    model_name = "Simple_ANN_PyTorch_Regressor"
    min_samples_for_valid_split = 100

    def __init__(
        self,
        D: Optional[int] = None,
        lr: float = 1e-3,
        activation: str = "tanh",
        optimizer_name: str = "adam",
    ) -> None:
        """
        Construct a new regressor.

        Args:
            D (int, optional): Size of the input layer. Defaults to None (set in `fit`).
            lr (float, optional): Learning rate for optimizer. Defaults to 1e-3.
            activation (str, optional): Activation function for hidden layer.
                                Defaults to "relu". Options: ["relu", "tanh", "none"]
            optimizer_name (str, optional): Optimizer for the loss function.
                                Defaults to "adam". Options: ["adam", "sgd", "rmsprop"]
        """
        self.D = D
        self.activation = activation
        self.lr = lr
        self._print_period = 10
        self.optimizer_name = optimizer_name
        # following are set when fitting to data
        self.net = None
        self.criterion = None
        self.optimizer = None

    def _build_model(self):
        """
        Build and set up the model, loss function, and optimizer.
        """
        self.net = Net(D=self.D, activation=self.activation).to(device)
        self.criterion = T.nn.MSELoss()
        self.optimizer = get_optimizer(
            self.optimizer_name, self.net.parameters(), lr=self.lr
        )

    def fit(
        self,
        train_inputs: pd.DataFrame,
        train_targets: pd.Series,
        batch_size: int = 64,
        epochs: int = 1000,
        verbose: int = 1,
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Fit the model to the given training data.

        Args:
            train_inputs (pd.DataFrame): Training inputs.
            train_targets (pd.Series): Training targets.
            batch_size (int, optional): Batch size for training. Defaults to 64.
            epochs (int, optional): Number of epochs to train. Defaults to 1000.
            verbose (int, optional): Whether to print training progress.
                                    Defaults to 1.

        Returns:
            List[Dict[str, Union[int, float]]]: Training losses for each epoch.
        """
        N, self.D = train_inputs.shape
        self._build_model()

        if N >= self.min_samples_for_valid_split:
            train_X, valid_X, train_y, valid_y = train_test_split(
                train_inputs.values,
                train_targets,
                test_size=0.2,
                random_state=42,
            )
        else:
            train_X, valid_X, train_y, valid_y = train_inputs, None, train_targets, None

        train_dataset = CustomDataset(train_X, train_y)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

        if valid_X is not None and valid_y is not None:
            valid_dataset = CustomDataset(valid_X, valid_y)
            valid_loader = DataLoader(
                dataset=valid_dataset, batch_size=batch_size, shuffle=True
            )
        else:
            valid_loader = None
        losses = self._run_training(
            train_loader,
            valid_loader,
            epochs,
            use_early_stopping=True,
            patience=30,
            verbose=verbose,
        )

        return losses

    def _run_training(
        self,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader],
        epochs: int,
        use_early_stopping: bool = True,
        patience: int = 30,
        verbose: int = 1,
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Run the training loop.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            valid_loader (Optional[DataLoader]): DataLoader for validation data.
            epochs (int): Number of epochs to train.
            use_early_stopping (bool, optional): Whether to use early stopping.
                            Defaults to True.
            patience (int, optional): Number of epochs to wait before stopping
                            training when validation loss doesn't decrease.
                            Defaults to 3.
            verbose (int, optional): Whether to print training progress.
                            Defaults to 1.

        Returns:
            List[Dict[str, Union[int, float]]]: Training losses for each epoch.
        """
        best_loss = 1e7
        losses = []
        trigger_times = 0
        for epoch in range(epochs):
            for times, data in enumerate(train_loader):
                inputs, labels = data[0].to(device).float(), data[1].to(device).float()
                output = self.net(inputs)
                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # current_loss = loss.item()
            train_loss = get_loss(self.net, device, train_loader, self.criterion)
            epoch_log = {"epoch": epoch, "train_loss": train_loss}

            if valid_loader is not None:
                val_loss = get_loss(self.net, device, valid_loader, self.criterion)
                epoch_log["val_loss"] = val_loss

            # Show progress
            if verbose == 1:
                if epoch % self._print_period == 0 or epoch == epochs - 1:
                    val_loss_str = (
                        ""
                        if valid_loader is None
                        else f", val_loss: {np.round(val_loss, 5)}"
                    )
                    logger.info(
                        f"Epoch: {epoch+1}/{epochs}"
                        f", loss: {np.round(train_loss, 5)}"
                        f"{val_loss_str}"
                    )

            losses.append(epoch_log)

            if use_early_stopping:
                # Early stopping
                if valid_loader is not None:
                    current_loss = val_loss
                else:
                    current_loss = train_loss

                if current_loss < best_loss:
                    trigger_times = 0
                    best_loss = current_loss
                else:
                    trigger_times += 1
                    if trigger_times >= patience:
                        if verbose == 1:
                            logger.info("Early stopping!")
                        return losses

        return losses

    def predict(self, inputs: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels for the given data.

        Args:
            inputs (Union[pd.DataFrame, np.ndarray]): The input data.

        Returns:
            np.ndarray: The predicted class labels.
        """
        if isinstance(inputs, pd.DataFrame):
            X = inputs.values
        else:
            X = inputs
        X = T.tensor(X, dtype=T.float32).to(device)
        preds = np.squeeze(self.net(X).detach().cpu().numpy())
        if np.ndim(preds) == 0:
            preds = np.reshape(preds, [1])
        return preds

    def summary(self):
        """
        Print a summary of the neural network.
        """
        self.net.summary()

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Evaluate the model and return the R-squared value.

        Args:
            x_test (pd.DataFrame): Training inputs.
            y_test (pd.Series): Training targets.

        Returns:
            float: R-squared value of the model on test data.

        Raises:
            NotFittedError: If the model is not fitted yet.
        """
        if self.net is not None:
            dataset = CustomDataset(x_test, y_test)
            data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

            # Get predictions and actual values
            predictions, actuals = [], []
            self.net.eval()
            with T.no_grad():
                for data in data_loader:
                    inputs = data[0].to(device).float()
                    targets = data[1].to(device).float()
                    output = self.net(inputs.view(inputs.shape[0], -1))
                    predictions.extend(output.cpu().numpy())
                    actuals.extend(targets.cpu().numpy())

            predictions = np.array(predictions)
            actuals = np.array(actuals)

            # Calculate R-squared
            ssr = np.sum((predictions - actuals) ** 2)
            sst = np.sum((actuals - actuals.mean()) ** 2)
            r_squared = 1 - (ssr / sst)

            return r_squared

        else:
            raise NotFittedError("Model is not fitted yet.")

    def save(self, model_path: str):
        """
        Save the model to the specified path.

        Args:
            model_path (str): Path to save the model.

        Raises:
            NotFittedError: If the model is not fitted yet.
        """
        if self.net is None:
            raise NotFittedError("Model is not fitted yet.")
        model_params = {
            "D": self.D,
            "activation": self.activation,
            "lr": self.lr,
            "optimizer_name": self.optimizer_name,
        }
        joblib.dump(model_params, os.path.join(model_path, MODEL_PARAMS_FNAME))
        T.save(self.net.state_dict(), os.path.join(model_path, MODEL_WTS_FNAME))

    @classmethod
    def load(cls, model_path: str) -> "Regressor":
        """
        Load the model from the specified path.

        Args:
            model_path (str): Path to load the model from.

        Returns:
            Regressor: The loaded model.
        """
        model_params = joblib.load(os.path.join(model_path, MODEL_PARAMS_FNAME))
        regressor_model = cls(**model_params)
        regressor_model._build_model()
        regressor_model.net.load_state_dict(
            T.load(os.path.join(model_path, MODEL_WTS_FNAME))
        )
        return regressor_model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.model_name} ("
            f"D: {self.D}, "
            f"activation: {self.activation}, "
            f"lr: {self.lr}, "
            f"optimizer_name: {self.optimizer_name})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Regressor:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data targets.
        hyperparameters (dict): Hyperparameters for the regressor.

    Returns:
        'Regressor': The regressor model
    """
    regressor = Regressor(**hyperparameters)
    regressor.fit(train_inputs=train_inputs, train_targets=train_targets)
    return regressor


def predict_with_model(regressor: Regressor, data: pd.DataFrame) -> np.ndarray:
    """
    Predict regression targets for the given data.

    Args:
        regressor (Regressor): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        np.ndarray: The predicted regression targets.
    """
    return regressor.predict(data)


def save_predictor_model(model: Regressor, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Regressor:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Regressor, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the regressor model and return the r-squared value.

    Args:
        model (Regressor): The regressor model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The targets of the test data.

    Returns:
        float: The r-sq value of the regressor model.
    """
    return model.evaluate(x_test, y_test)

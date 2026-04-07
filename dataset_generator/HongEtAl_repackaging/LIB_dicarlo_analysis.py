import copy
import numpy as np
import random
import os
from datetime import datetime
from PIL import Image
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer


# <editor-fold desc="DATASET MANIPULATION">
def group_average(dataset_original, coordinate, across, over=None):
    """
    returns a dataset where values are averaged for elements having the same subcoordinate "across"

    Parameters
    ----------
    dataset: dict
        The dataset, in the form of a python dictionary.

    coordinate: str
        The coordinate to which the subccordinate belongs

    across: str
        the subcoordinate

    over: iterable(str)
        list of strings, containing subcoordinates that we want to remove.

    Returns
    -------
    averaged_dataset: dict
        the averaged dataset
    """

    if over is None:
        over = []

    dataset = copy.deepcopy(dataset_original)

    # group the subcoordinates by those that have the same value across the subcoordinate "across"

    # determine where same elements are in the "across" coordinate
    _, unique_indices, inverse_indices = np.unique(dataset[coordinate][across],
                                                               return_index=True, return_inverse=True)
    # create a list of lists, each sublist contains indices corresponding to values of the
    # subcoordinate across that are the same
    groups_indices_list = [np.where(inverse_indices == idx)[0] for idx in range(len(unique_indices))]

    # remove "over" subcoordinates, if any
    for ovr in over:
        del dataset[coordinate][ovr]

    # reduce the subcoordinates to a single representative for each group
    for key in dataset[coordinate].keys():
        dataset[coordinate][key] = dataset[coordinate][key][unique_indices]

    # compute the average within each subgroup. Create a new values array with the averages
    # retrieve to which index "coordinate" corresponds
    coordinate_index = dataset["coordinate_positions"].index(coordinate)

    # construct the reduced (averaged) values
    values_ndim = dataset["values"].ndim
    reduced_values_list = []
    for group_indices in groups_indices_list:
        # reduce array to the group indices, along the "coordinate" dimension
        # Create a tuple of slices
        slices = [slice(None)] * values_ndim
        slices[coordinate_index] = group_indices
        # Convert slices to a tuple
        slices = tuple(slices)
        # Reduce the array to the specified indices along the specified dimension
        group_values = dataset["values"][slices]
        # average
        group_mean = np.mean(group_values, axis=coordinate_index)
        # append
        reduced_values_list.append(group_mean)

    # create the new values array by stacking the averaged values along the coordinate dimension
    # (i.e. recreating the coordinate dimension)
    dataset["values"] = np.stack(reduced_values_list, axis=coordinate_index)

    return dataset


def select(dataset_original, coordinate, indices):

    dataset = copy.deepcopy(dataset_original)

    # SELECT THE VALUES
    # retrieve to which index "coordinate" corresponds
    coordinate_index = dataset["coordinate_positions"].index(coordinate)
    # create indices to select the values at the appropriate coordinate
    values_ndim = dataset["values"].ndim
    # Create a tuple of slices
    slices = [slice(None)] * values_ndim
    slices[coordinate_index] = indices
    # Convert slices to a tuple
    slices = tuple(slices)
    # Reduce the array to the specified indices along the specified dimension
    dataset["values"] = dataset["values"][slices]

    # SELECT THE COORDINATE
    for key in dataset[coordinate].keys():
        dataset[coordinate][key] = dataset[coordinate][key][indices]

    return dataset
# </editor-fold>


# <editor-fold desc="UTILITY">
def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def unseed_everything():
    # set a random seed using the current time
    seed = int(datetime.now().timestamp())
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True


def load_pixel_layer_dataset(folder_path, responses_dataset):
    # folder_path is where the images are stored, e.g. ~/.brainio/image_dicarlo_hvm
    # load images and convert to numpy array
    image_list = []
    for filename in responses_dataset["presentation"]["image_file_name"]:
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale for extra safety
        img_array = np.array(img).flatten()  # Flatten the image to 1D array
        image_list.append(img_array)
    image_array = np.array(image_list)

    # create a dataset dictionary, just like the one for the neural responses
    pixel_layer_dataset = copy.deepcopy(responses_dataset)
    # replace neural responses with pixels
    pixel_layer_dataset["values"] = np.transpose(image_array)
    # transposing so the order is the same as in the responses dataset
    # redefine the neuroid coordinate (it is a "pixelid" coordinate now)
    pixel_layer_dataset["neuroid"] = {"neuroid_id": np.array([str(i) for i in range(image_array.shape[1])])}
    # just assigning to each pixel an ID corresponding to the pixel number
    # the "presentation" coordinate will stay the same, since the images are the same

    return pixel_layer_dataset

# </editor-fold>


# Custom scoring function
def pearson_corr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def regression(X, y, number_neurons=128, number_random_subsamples=100, number_performance_crossvalidations=50,
               performance_crossvalidation_test_fraction=0.2, number_l2_crossvalidations=10, l2_strengths=None,
               verbose=False):
    """
    Performs the ridge regression analysis. The analysis goes on three loops, in nesting order:
    1. Loop over different subsets o neurons
    2. Cross-validation loop for evaluating performance
    3. Cross-validation loop for estimating the optimal l2 strength

    Parameters
    ----------
    X: ndarray
        Dataset features of shape [# of examples, # of features]

    y: ndarray
        Dataset labels of shape [# of examples]

    number_neurons: int
        Number of the neurons (i.e. features) to use a subset for the task

    number_random_subsamples: int
        number of random subsets of neurons on which to cross-validate the performance

    number_performance_crossvalidations: int

    performance_crossvalidation_test_fraction: float
        fraction of the dataset to use as test set.

    number_l2_crossvalidations: int
        number of subsplits of the train set in order to select the optimal l2 regularization strength.

    l2_strengths: iterable(float)
        list of l2 strengths among which to select the optimal one.

    verbose: bool
        If true, prints an update on the trainign status

    Returns
    -------
    results: dict
        Dictionary containing the regression results and parameters.
        See at the end of this function for the dictionary entries
    """
    if l2_strengths is None:
        l2_strengths = [0.01, 0.1, 1, 10, 100]

    # SETUP THE MODEL

    # Define the model (regression)
    model = Ridge()

    # Define the l2 strengths grid
    param_grid = {'alpha': l2_strengths}
    # Note: alpha is the name of the l2 regularization strength for Ridge regression

    # Custom scoring function
    scorer = make_scorer(pearson_corr, greater_is_better=True)

    # LOOP 1: RANDOM SUBSET OF NEURONS
    performances_subsets = []
    best_l2_strengths_subsets = []

    for subset in range(number_random_subsamples):
        if verbose:
            print(f"running subset {subset+1}/{number_random_subsamples}")

        # Select a random subset of neurons
        total_number_neurons = X.shape[1]
        subsample_indices = np.random.choice(total_number_neurons, number_neurons, replace=False)
        # replace=False avoids selecting the same index twice
        X = X[:, subsample_indices]

        # LOOP 2: PERFORMANCE CROSS-VALIDATION

        performances_crossval = []
        best_l2_strengths_crossval = []

        for _ in range(number_performance_crossvalidations):
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=performance_crossvalidation_test_fraction,
                                                                random_state=np.random.randint(1, 10000000))

            # LOOP 3: Inner cross-validation for hyperparameter tuning
            inner_cv = KFold(n_splits=number_l2_crossvalidations, shuffle=True,
                             random_state=np.random.randint(1, 10000000))
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring=scorer)
            grid_search.fit(X_train, y_train)

            # Save the best parameter for this split
            best_l2_strengths_crossval.append(grid_search.best_params_['alpha'])
            # END OF LOOP 3

            # Evaluate the best model on the test set
            best_model = grid_search.best_estimator_
            predictions = best_model.predict(X_test)
            score = pearson_corr(y_test, predictions)
            performances_crossval.append(score)

        # append the results from this subset
        performances_subsets.append(performances_crossval)
        best_l2_strengths_subsets.append(best_l2_strengths_crossval)

    # Convert lists to numpy arrays for convenience
    performances = np.array(performances_subsets)
    best_l2_strengths = np.array(best_l2_strengths_subsets)

    # create a dictionary with the results and return it
    results = {
        # results
        "performance": performances,
        "l2_strength": best_l2_strengths,
        # mean and std of results
        "performance_mean": np.mean(performances),
        "performance_std": np.std(performances),
        "l2_strength_mean": np.mean(best_l2_strengths),
        "l2_strength_std": np.std(best_l2_strengths),
        # params
        "number_neurons": number_neurons,
        "number_random_subsamples": number_random_subsamples,
        "number_performance_crossvalidations": number_performance_crossvalidations,
        "performance_crossvalidation_test_fraction": performance_crossvalidation_test_fraction,
        "number_l2_crossvalidations": number_l2_crossvalidations,
        "l2_strengths": l2_strengths
    }

    return results


def angular_regression(X, angle, number_neurons=128, number_random_subsamples=100, number_performance_crossvalidations=50,
                       performance_crossvalidation_test_fraction=0.2, number_l2_crossvalidations=10, l2_strengths=None,
                       verbose=False):
    """
    Performs the ridge regression analysis, regressing the sin and cos of an angular variable.
    The analysis goes on three loops, in nesting order:
    1. Loop over different subsets o neurons
    2. Cross-validation loop for evaluating performance
    3. Cross-validation loop for estimating the optimal l2 strength

    Parameters
    ----------
    X: ndarray
        Dataset features of shape [# of examples, # of features]

    angle: ndarray
        Dataset labels of shape [# of examples]. The labels should be angles in radians.

    number_neurons: int
        Number of the neurons (i.e. features) to use a subset for the task

    number_random_subsamples: int
        number of random subsets of neurons on which to cross-validate the performance

    number_performance_crossvalidations: int

    performance_crossvalidation_test_fraction: float
        fraction of the dataset to use as test set.

    number_l2_crossvalidations: int
        number of subsplits of the train set in order to select the optimal l2 regularization strength.

    l2_strengths: iterable(float)
        list of l2 strengths among which to select the optimal one.

    verbose: bool
        If true, prints an update on the trainign status

    Returns
    -------
    results_cosine: dict
        Dictionary containing the regression results and parameters, for fitting the cosine of the angle.
        Same dictionary returned by the "regression" function.
    results_sine: dict
        Dictionary containing the regression results and parameters, for fitting the sine of the angle.
        Same dictionary returned by the "regression" function.
    """

    # FIT COSINE
    # EXTRACT THE REGRESSION VARIABLE
    y = np.cos(angle)
    # RUN THE REGRESSION
    results_cosyne = regression(X, y, number_neurons=number_neurons, number_random_subsamples=number_random_subsamples,
                                number_performance_crossvalidations=number_performance_crossvalidations,
                                performance_crossvalidation_test_fraction=performance_crossvalidation_test_fraction,
                                number_l2_crossvalidations=number_l2_crossvalidations,
                                l2_strengths=l2_strengths, verbose=verbose)
    # FIT SINE
    # EXTRACT THE REGRESSION VARIABLE
    y = np.sin(angle)
    # RUN THE REGRESSION
    results_sine = regression(X, y, number_neurons=number_neurons, number_random_subsamples=number_random_subsamples,
                              number_performance_crossvalidations=number_performance_crossvalidations,
                              performance_crossvalidation_test_fraction=performance_crossvalidation_test_fraction,
                              number_l2_crossvalidations=number_l2_crossvalidations,
                              l2_strengths=l2_strengths, verbose=verbose)
    # </editor-fold>

    return results_cosyne, results_sine











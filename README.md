# Code Structure:

## src:
* `data_handler.py`:                    Class that loads and processes data
* `form_train_val_test_tf_batches.py`:  Form training, validation and test batches
                                        loaded data using Tensorflow's Dataset
                                        API
* `nn_.py`:                             The neural network
* `loss_and_relative_errors.py`:        Functionals that form the overall loss
                                        functional
* `optimize_.py`:                       The optimization routine for the neural network
* `metrics_.py`:                        Metrics class storing and updating the optimization information
* `get_hyperparameter_combinations.py`: Form combinations of the hyperparameters
                                        for scheduled training
* `schedule_and_run.py`:                Uses hyperparameter permutations to run a distributed
                                        schedule of training routines using mpi4py

## projects: Contains project specific wrappers and routines
* `training_driver_.py`:           Drives the training routine. Consists of the
                                   Hyperparameter class and calls the FilePaths class and the training_routine
                                   method
* `hyperparameter_optimizer_.py`:  Drives hyperparameter optimization for
                                   training neural networks. Utilizes scikit-optimize`s
                                   Bayesian optimization routines.
* `prediction_driver_.py`:         Drives the prediction routine given a trained neural
                                   network
* `plotting_driver_.py`:           Drives the plotting routine given a trained neural
                                   network and predictions
* `scheduler_training_.py`:        Drives the formation of hyperparameter permutations
                                   and schedule of training routines using mpi4py
* utils_project:
	* `file_paths_.py`:        Specifies the file paths for the data, trained
                               neural network, predictions and plots
	* `training_routine_.py`:  Loads the data, constructs the neural
                               network and runs the optimization routine
	* `hyperparameter_optimization_training_routine_*.py`: Optimization
                               routine for Bayesian hyperparameter
                               optimization
	* `predict_and_save_.py`:  Prediction routine; using trained network,
                               form and save predictions given trained
                               neural network
	* `plot_and_save_.py`:     Plotting routine; using trained network
                               and predictions, plot predictiions and
                               optimization information

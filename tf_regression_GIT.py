from tf_regression_functions import *
from ray import tune
from ray import init as parallel_init
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parallel_init(log_to_driver=False)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Dataset Name and Omega_m list
datasets = ['dataset_testB1.5_0.01.txt']
omega_ms = [0.2, 0.3, 0.4]

# Configuration for every trained model in the bootstrap procedure
sim_config = {'dataset': tune.grid_search(datasets),
              'current_dir': current_dir,
              'learning_rate': 4E-5,
              'epochs': 800,
              'omega_m': tune.grid_search(omega_ms)}

# Number of bootstrap iterations
n_boot = 30

# Run parallel processes
tune.run(tune.with_parameters(omega_training,
                              save_ris=True,
                              bootstrap=True),
         num_samples=n_boot,
         resources_per_trial={'cpu': 1, 'gpu': .20},
         config=sim_config)

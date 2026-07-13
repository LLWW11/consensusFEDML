import fedml
from fedml import FedMLRunner
from fedml.constants import (
    FEDML_TRAINING_PLATFORM_SIMULATION,
    FEDML_SIMULATION_TYPE_SP,
)
import init_test
import runner_test
import mnist_dataloader

if __name__ == "__main__":
    fedml._global_training_type = FEDML_TRAINING_PLATFORM_SIMULATION  # 'simulation'
    fedml._global_comm_backend = FEDML_SIMULATION_TYPE_SP  # 'sp

    # init FedML framework
    args = init_test.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = mnist_dataloader.load_data(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = runner_test.FedMLRunner(args, device, dataset, model)
    fedml_runner.run()

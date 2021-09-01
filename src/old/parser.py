import yaml


class ParsedConfig:

    def __init__(self, yaml_file):
        with open(yaml_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.batch_size = config.get('BATCH_SIZE')
        self.optimizer = config.get('OPTIMIZER')
        self.initial_learning_rate = config.get('INITIAL_LEARNING_RATE')
        self.final_learning_rate = config.get('FINAL_LEARNING_RATE')
        self.num_epochs = config.get('NUM_EPOCHS')
        self.model = config.get('MODEL')

    # def _load_config(self):
    #     try:
    #         config_file = open(self._yaml_file, 'r')
    #
    #     except FileNotFoundError:
    #         print(f'[WARNING] config file {self._yaml_file} not found! Setting up default config...')
    #         raise FileNotFoundError
    #
    #     config = yaml.load(config_file)
    #     raise NotImplementedError
    #
    # def _load_default(self):
    #     self.batch_size = 128
    #     self.regression = True
    #     self.binning = False
    #     self.weighted_sampling = False
    #     self.num_workers = 8
    #
    #     # model
    #     self.batch_norm = False
    #
    #     # training
    #     self.optimizer = 'adam'
    #     self.dropout = 0.8
    #     self.initial_learning_rate = 1e-3
    #     self.final_learning_rate = 1e-4
    #     self.loss_fn = 'mse'
    #     self.num_epochs = 100

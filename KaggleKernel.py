class KaggleKernel:
    def __init__(self):
        self.model = None

        self.train_X_all = None
        self.train_y_all = None
        self.train_X = None
        self.train_y = None

        self.train_y_aux = None
        self.train_y_aux_all = None
        self.train_y_identity = None
        self.train_X_identity = None

        self.judge = None  # for analyze the result
        pass

    def build_model(self):
        pass

    def set_loss(self):
        pass

    def set_model(self):
        pass

    def set_metrics(self):
        pass

    def prepare_train_data(self):
        pass

    def prepare_dev_data(self):
        pass

    def prepare_test_data(self):
        pass

    def save_result(self):
        pass

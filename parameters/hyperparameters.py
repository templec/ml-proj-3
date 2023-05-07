from custom_enums import FileOption


class HyperParameters():
    def __init__(self, args_get=False, **kwargs):
        self.num_epochs = None
        self.batch_size = None
        self.lr = None
        self.num_workers = None
        self.train_path = None
        self.pretrained_model_path = None
        self.save_every_num_epoch = None

        self.accuracy_loss_dict = {}

        if args_get:
            self.get_from_args(**kwargs)
        else:
            self.get_from_custom(**kwargs)

    def set_accuracy_loss_dict(self, file_option_list: list, file_option: FileOption):
        self.accuracy_loss_dict[file_option] = file_option_list

    def add_accuracy_loss_dict_value(self, file_option: FileOption, file_option_value):
        self.accuracy_loss_dict[file_option].append(file_option_value)

    def get_accuracy_loss_dict_list(self, file_option: FileOption):
        return self.accuracy_loss_dict[file_option]

    def print_accuracy_loss_dict(self):
        print(f"---------------------------------------------------")
        print(f"accuracy/loss of training and validation")

        for file_option in FileOption:
            print(f"{file_option.name}: {self.accuracy_loss_dict[file_option]}")

        print(f"---------------------------------------------------")

    def get_from_args(self, **kwargs):
        # args, train_path

        # self.num_epochs = args.epochs
        # self.batch_size = args.batch_size
        # self.lr = args.lr
        # self.num_workers = args.num_workers
        # self.train_path = train_path
        # self.pretrained_model = args.pretrained_model_path

        args = kwargs.get("0")
        train_path = kwargs.get("1")
        save_every_num_epoch = kwargs.get("2")

        self.num_epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.num_workers = args.num_workers
        self.train_path = train_path
        self.pretrained_model_path = args.pretrained_model_path
        self.save_every_num_epoch = save_every_num_epoch

    def get_from_custom(self, **kwargs):
        # self, num_epochs,
        #                         batch_size,
        #                         lr,
        #                         num_workers,
        #                         train_path,
        #                         pretrained_model

        # self.num_epochs = num_epochs
        # self.batch_size = batch_size
        # self.lr = lr
        # self.num_workers = num_workers
        # self.train_path = train_path
        # self.pretrained_model = pretrained_model

        self.num_epochs = kwargs.get("0")
        self.batch_size = kwargs.get("1")
        self.lr = kwargs.get("2")
        self.num_workers = kwargs.get("3")
        self.train_path = kwargs.get("4")
        self.pretrained_model_path = kwargs.get("5")
        self.save_every_num_epoch = kwargs.get("6")

    def print_parameters(self):
        print(f"num epochs: {self.num_epochs}")
        print(f"batch size: {self.batch_size}")
        print(f"learning rate: {self.lr}")
        print(f"num workers: {self.num_workers}")
        print(f"training path: {self.train_path}")
        print(f"pretrained model path: {self.pretrained_model_path}")

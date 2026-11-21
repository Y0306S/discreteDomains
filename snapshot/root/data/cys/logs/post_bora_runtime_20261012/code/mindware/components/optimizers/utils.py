from ConfigSpace import Configuration


class MyConfiguration(Configuration):

    def __init__(self, configuration: Configuration, extra_info: dict = None):
        super().__init__(configuration.configuration_space, configuration.get_dictionary())

        self.extra_info = extra_info if extra_info is not None else {}

    def __str__(self):
        config_str = super().__str__()
        return f"{config_str}, extra_info: {self.extra_info}"

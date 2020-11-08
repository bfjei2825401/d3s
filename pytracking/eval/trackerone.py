import importlib


class TrackerOne:
    def __init__(self, name: str, parameter_name: str):
        self.name = name
        self.parameter_name = parameter_name

        tracker_module = importlib.import_module('pytracking.tracker.{}'.format(self.name))
        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(self.name, self.parameter_name))
        self.parameters = param_module.parameters()
        self.tracker_class = tracker_module.get_tracker_class()
        self.tracker = self.tracker_class(self.parameters)

    def init(self, image, box):
        self.tracker.initialize(image, box)

    def update(self, image):
        return self.tracker.track(image)

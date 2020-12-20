import numpy as np
from got10k.experiments import ExperimentGOT10k
from got10k.trackers import Tracker as EvalTracker
import importlib


class SegmEvalTracker(EvalTracker):
    def __init__(self, name: str, parameter_name: str, tracker_module_name: str):
        super(SegmEvalTracker, self).__init__(name=name)
        self.name = name
        self.parameter_name = parameter_name

        tracker_module = importlib.import_module('pytracking.tracker.{}'.format(tracker_module_name))

        param_module = importlib.import_module(
            'pytracking.parameter.{}.{}'.format(tracker_module_name, self.parameter_name))
        self.parameters = param_module.parameters()
        self.tracker_class = tracker_module.get_tracker_class()
        self.tracker = self.tracker_class(self.parameters)

    def init(self, image, box):
        self.tracker.initialize(np.asarray(image), box)

    def update(self, image):
        return self.tracker.track(np.asarray(image))


def main():
    trackers = [SegmEvalTracker('SegmSKMeanMaxEp0021_got10k_up3', 'default_params_ep0021', 'segm_sk_meanmax_up')]

    # run experiments on GOT-10k (validation subset)
    experiment = ExperimentGOT10k('/home/slz/GitHub/d3s/dataset/GOT10k',
                                  result_dir='results',
                                  subset='test',
                                  report_dir='reports')
    for tracker in trackers:
        experiment.run(tracker, visualize=False)

    # report performance
    experiment.report([tracker.name for tracker in trackers])


if __name__ == '__main__':
    main()

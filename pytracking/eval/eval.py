import numpy as np
from got10k.experiments import ExperimentGOT10k
from got10k.trackers import Tracker as EvalTracker

from pytracking.eval.trackerone import TrackerOne


class D3SEvalTracker(EvalTracker):
    def __init__(self, tracker, name):
        super(D3SEvalTracker, self).__init__(name=name)
        self.tracker = tracker

    def init(self, image, box):
        self.tracker.init(np.asarray(image), box)

    def update(self, image):
        return self.tracker.update(np.asarray(image))


def main():
    # setup tracker
    d3s = TrackerOne('segm', 'default_params')
    SegmSK = TrackerOne('segm_sk', 'finetune_params')
    trackers = [D3SEvalTracker(d3s, 'D3S'), D3SEvalTracker(SegmSK, 'SegmSK')]

    # run experiments on GOT-10k (validation subset)
    experiment = ExperimentGOT10k('/home/slz/GitHub/d3s/dataset/GOT10k', result_dir='eval/results',
                                  report_dir='eval/reports')
    for tracker in trackers:
        experiment.run(tracker, visualize=False)

    # report performance
    experiment.report([tracker.name for tracker in trackers])


if __name__ == '__main__':
    main()

import importlib
import os
import pickle
from pytracking.evaluation.environment import env_settings

from .tracker import Tracker


class TrackerEp(Tracker):
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
    """

    def __init__(self, name: str, parameter_name: str, ep_num: int, run_id: int = None):
        self.ep_num = ep_num
        self.name = name
        self.parameter_name = parameter_name
        self.run_id = run_id
        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}{:04d}'.format(env.results_path, self.name, self.parameter_name, self.ep_num)
        else:
            self.results_dir = '{}/{}/{}_ep{:04d}_{:03d}'.format(env.results_path, self.name, self.parameter_name,
                                                                 self.ep_num, self.run_id)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        tracker_module = importlib.import_module('pytracking.tracker.{}'.format(self.name))

        self.parameters = self.get_parameters()
        self.tracker_class = tracker_module.get_tracker_class()

        self.default_visualization = getattr(self.parameters, 'visualization', False)
        self.default_debug = getattr(self.parameters, 'debug', 0)

    def get_parameters(self):
        """Get parameters."""

        parameter_file = '{}/parameters.pkl'.format(self.results_dir)
        if os.path.isfile(parameter_file):
            return pickle.load(open(parameter_file, 'rb'))

        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(self.name, self.parameter_name))
        params = param_module.parameters(self.ep_num)

        if self.run_id is not None:
            pickle.dump(params, open(parameter_file, 'wb'))

        return params

from pytracking.evaluation import Tracker, OTBDataset, NFSDataset, UAVDataset, TPLDataset, VOTDataset, \
    TrackingNetDataset, LaSOTDataset, GOT10KDatasetTest

from pytracking.evaluation.tracker_ep import TrackerEp


def otb_all_ep():
    trackers = [TrackerEp('segm_sk3x3_max', 'default_params_ep', i) for i in range(39, 41)]
    dataset = OTBDataset()
    return trackers, dataset
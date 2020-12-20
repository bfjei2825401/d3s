from pytracking.evaluation import Tracker, OTBDataset, NFSDataset, UAVDataset, TPLDataset, VOTDataset, \
    TrackingNetDataset, LaSOTDataset, GOT10KDatasetTest

from pytracking.evaluation.tracker_ep import TrackerEp


def trackingnet_ep0026():
    trackers = [TrackerEp('segm_sk_max', 'default_params_ep', i) for i in range(26, 27)]
    dataset = TrackingNetDataset()
    return trackers, dataset

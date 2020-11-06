from pytracking.evaluation import Tracker, OTBDataset, NFSDataset, UAVDataset, TPLDataset, VOTDataset, \
    TrackingNetDataset, LaSOTDataset, GOT10KDatasetTest


def segm_sk_test():
    trackers = [Tracker('segm_sk', 'default_params', i) for i in range(1)]
    dataset = LaSOTDataset() + TrackingNetDataset() + GOT10KDatasetTest()
    return trackers, dataset

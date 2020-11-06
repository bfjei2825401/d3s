from pytracking.evaluation import Tracker, OTBDataset, NFSDataset, UAVDataset, TPLDataset, VOTDataset, \
    TrackingNetDataset, LaSOTDataset, GOT10KDatasetTest


def segm_test():
    trackers = [Tracker('segm', 'default_params', i) for i in range(1)]
    dataset = OTBDataset() + UAVDataset() + LaSOTDataset() + TrackingNetDataset() + GOT10KDatasetTest()
    return trackers, dataset

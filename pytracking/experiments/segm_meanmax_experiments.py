from pytracking.evaluation import Tracker, OTBDataset, NFSDataset, UAVDataset, TPLDataset, VOTDataset, \
    TrackingNetDataset, LaSOTDataset, GOT10KDatasetTest


def segm_meanmax_test_all():
    trackers = [Tracker('segm_sk_meanmax', 'default_params_ep0021', i) for i in range(1)]
    dataset = OTBDataset() + UAVDataset() + LaSOTDataset() + TrackingNetDataset() + GOT10KDatasetTest()
    return trackers, dataset

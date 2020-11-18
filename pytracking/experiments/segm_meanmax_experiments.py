from pytracking.evaluation import Tracker, OTBDataset, NFSDataset, UAVDataset, TPLDataset, VOTDataset, \
    TrackingNetDataset, LaSOTDataset, GOT10KDatasetTest


def all():
    trackers = [Tracker('segm_sk_meanmax', 'default_params_ep0021', i) for i in range(1)]
    dataset = OTBDataset() + UAVDataset() + LaSOTDataset() + TrackingNetDataset() + GOT10KDatasetTest()
    return trackers, dataset


def otb():
    trackers = [Tracker('segm_sk_meanmax', 'default_params_ep0021', 1)]
    dataset = OTBDataset()
    return trackers, dataset


def lasot():
    trackers = [Tracker('segm_sk_meanmax', 'default_params_ep0021', 2)]
    dataset = LaSOTDataset()
    return trackers, dataset


def got10k_test():
    trackers = [Tracker('segm_sk_meanmax', 'default_params_ep0021', 3)]
    dataset = GOT10KDatasetTest()
    return trackers, dataset


def trackingnet():
    trackers = [Tracker('segm_sk_meanmax', 'default_params_ep0021', 4)]
    dataset = TrackingNetDataset()
    return trackers, dataset


def uav():
    trackers = [Tracker('segm_sk_meanmax', 'default_params_ep0021', 5)]
    dataset = UAVDataset()
    return trackers, dataset


def nfs():
    trackers = [Tracker('segm_sk_meanmax', 'default_params_ep0021', 6)]
    dataset = NFSDataset()
    return trackers, dataset

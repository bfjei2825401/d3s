from pytracking.evaluation import Tracker, OTBDataset, NFSDataset, UAVDataset, TPLDataset, VOTDataset, \
    TrackingNetDataset, LaSOTDataset, GOT10KDatasetTest


def all():
    trackers = [Tracker('segm', 'default_params', i) for i in range(1)]
    dataset = OTBDataset() + UAVDataset() + LaSOTDataset() + TrackingNetDataset() + GOT10KDatasetTest()
    return trackers, dataset


def otb():
    trackers = [Tracker('segm', 'default_params', 1)]
    dataset = OTBDataset()
    return trackers, dataset


def lasot():
    trackers = [Tracker('segm', 'default_params', 2)]
    dataset = LaSOTDataset()
    return trackers, dataset


def got10k_test():
    trackers = [Tracker('segm', 'default_params', 3)]
    dataset = GOT10KDatasetTest()
    return trackers, dataset


def trackingnet():
    trackers = [Tracker('segm', 'default_params', 4)]
    dataset = TrackingNetDataset()
    return trackers, dataset


def uav():
    trackers = [Tracker('segm', 'default_params', 5)]
    dataset = UAVDataset()
    return trackers, dataset


def nfs():
    trackers = [Tracker('segm', 'default_params', 6)]
    dataset = NFSDataset()
    return trackers, dataset
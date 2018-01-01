
class SearchWindowParameters:
    def __init__(self, scale, weight, yrange, cellsPerStep, svmThreshold):
        self.scale = scale
        self.weight = weight
        self.yRange = yrange
        self.cellsPerStep = cellsPerStep
        self.svmThreshold = svmThreshold

class Parameters:

    class HistogramSettings:
        bins = 32
        range = (0,256)
    class HogSettings:
        orient = 9
        pix_per_cell = 8
        cell_per_block  = 2
        feature_vec = False

    class SpatialSettings:
        new_size = (32, 32)

    class SearchSettings:
        #y_start = 400
        #y_stop = 656
        #scale = [[1.75, 6], [1.5,3], [2,9], [1.25, 0.25]] #[1.25, 1.5, 1.75]
        min_hotspot = 5
        searchWindowList = []
        searchWindowList.append(SearchWindowParameters(scale=1.75,
                                                       weight=6,
                                                       yrange=(400, 656),
                                                       cellsPerStep=2,
                                                       svmThreshold=0))
        searchWindowList.append(SearchWindowParameters(scale=1.5,
                                                       weight=3,
                                                       yrange=(400, 656),
                                                       cellsPerStep=2,
                                                       svmThreshold=0))
        searchWindowList.append(SearchWindowParameters(scale=1.25,
                                                       weight=0.5,
                                                       yrange=(400, 500),
                                                       cellsPerStep=2,
                                                       svmThreshold=0.5))
        searchWindowList.append(SearchWindowParameters(scale=1,
                                                       weight=0.5,
                                                       yrange=(400, 500),
                                                       cellsPerStep=1,
                                                       svmThreshold=0.5))
        #searchWindowList.append(SearchWindowParameters(scale=0.75,
        #                                               weight=1,
        #                                               yrange=(400, 500),
        #                                               cellsPerStep=1,
        #                                               svmThreshold=0))
        threshold = 1.5
        averages = 15

    class Annotation:
        bounding_box_individual = False
        heatmap = False
        bounding_box_heatmap_average=True
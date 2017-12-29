
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
        y_start = 400
        y_stop = 656
        scale = [[1.75, 6],[1.5,3]] #[1.25, 1.5, 1.75]
        threshold = 2
        averages = 15

    class Annotation:
        bounding_box_individual = True
        heatmap = True
        bounding_box_heatmap_average=True
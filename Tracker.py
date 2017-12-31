from FeatureExtractor import *
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
import collections

class Tracker:
    def __init__(self, featureList, parameters, svc, scaler ):
        self.featureList = featureList
        self.parameters = parameters
        self.svc = svc
        self.scaler = scaler

        self.featureExtractor = FeatureExtractor(self.parameters, self.featureList)
        self.heatmapHistory = collections.deque(maxlen=parameters.SearchSettings.averages)


    def findBoundingBoxes(self, image):

        boundingBoxList = []

        img = image.astype(np.float32) / 255


        for searchWindowParameters in self.parameters.SearchSettings.searchWindowList:
            scale = searchWindowParameters.scale
            weight = searchWindowParameters.weight
            y_start = searchWindowParameters.yRange[0]
            y_stop = searchWindowParameters.yRange[1]
            cells_per_step = searchWindowParameters.cellsPerStep  # Instead of overlap, define how many cells to step
            svmThreshold = searchWindowParameters.svmThreshold

            img_tosearch = img[y_start:y_stop, :, :]
            ctrans_tosearch = FeatureExtractor.convert_color(img_tosearch, enum_ColorSpaces.ycrcb)

            if scale != 1:
                imshape = ctrans_tosearch.shape
                img_converted = cv2.resize(ctrans_tosearch,
                                           (np.int(imshape[1] / scale),
                                            np.int(imshape[0] / scale)))
            else:
                img_converted = img_tosearch

            # Define blocks and steps as above
            nxblocks = (img_converted.shape[1] // self.parameters.HogSettings.pix_per_cell) - self.parameters.HogSettings.cell_per_block + 1
            nyblocks = (img_converted.shape[0] // self.parameters.HogSettings.pix_per_cell) - self.parameters.HogSettings.cell_per_block + 1
            nfeat_per_block = self.parameters.HogSettings.orient * self.parameters.HogSettings.cell_per_block ** 2

            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = self.parameters.HogSettings.pix_per_cell**2
            nblocks_per_window = (window // self.parameters.HogSettings.pix_per_cell) - self.parameters.HogSettings.cell_per_block + 1

            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

            # Compute individual channel HOG features for the entire image
            hog_features_deep = self.featureExtractor.preCalculateHogFeatures(img_converted,
                                                                              self.parameters.HogSettings.orient,
                                                                              self.parameters.HogSettings.pix_per_cell,
                                                                              self.parameters.HogSettings.cell_per_block,
                                                                              feature_vec=False)

            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb * cells_per_step
                    xpos = xb * cells_per_step
                    # Extract HOG for this patch
                    hog_feat1 = hog_features_deep[0][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat2 = hog_features_deep[1][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat3 = hog_features_deep[2][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                    xleft = xpos * self.parameters.HogSettings.pix_per_cell
                    ytop = ypos * self.parameters.HogSettings.pix_per_cell

                    # Extract the image patch
                    selected_image = ctrans_tosearch[ytop:ytop + window, xleft:xleft + window]
                    if selected_image.shape != (window,window,3):
                        continue
                    subimg = cv2.resize(selected_image, (64, 64))

                    # Get color features
                    spatial_features = FeatureExtractor.bin_spatial(subimg, size=self.parameters.SpatialSettings.new_size)
                    hist_features = FeatureExtractor.color_hist(subimg, nbins=self.parameters.HistogramSettings.bins)

                    # Scale features and make a prediction
                    test_features = self.scaler.transform(
                        np.hstack((hog_features, spatial_features, hist_features)).reshape(1, -1))
                    # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                    test_prediction = self.svc.decision_function(test_features)

                    if test_prediction[0] > 0:
                        xbox_left = np.int(xleft * scale)
                        ytop_draw = np.int(ytop * scale)
                        win_draw = np.int(window * scale)

                        if test_prediction[0] < -0.5:
                            color = (255,0,0)
                        elif test_prediction < 0.5:
                            color = (0,255,0)
                        elif test_prediction < 1:
                            color = (0,0,255)
                        elif test_prediction < 1.5:
                            color = (255,255,0) #yellow
                        else:
                            color = (0,255,255) #purple


                        #draw the bouding box, but don't count in heatmap if value is less than threshold
                        if test_prediction[0] > searchWindowParameters.svmThreshold:
                            scaled_weight = weight*test_prediction[0]
                        else:
                            scaled_weight = 0



                        boundingBoxList.append(((xbox_left, ytop_draw + y_start),
                                                (xbox_left + win_draw, ytop_draw + win_draw + y_start), scaled_weight, color))

        return boundingBoxList

    def generateHeatMap(self, image, bounding_boxes, threshold=1):
        heatmap = np.zeros_like(image, dtype=np.float32)
        for box in bounding_boxes:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            weight = box[2]
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0], 0] += weight

        #Thanks to Alex_Cui for deque idea
        #https://discussions.udacity.com/t/how-to-take-average-heatmaps/242409/5

        self.heatmapHistory.append(heatmap)
        heatmap_average = np.average(self.heatmapHistory, axis=0)

        # Zero out pixels below the threshold
        heatmap_average[heatmap_average <= threshold] = 0
        # Return thresholded map

        heatmap_scaled = heatmap_average * 255 / np.max(heatmap_average)
        return heatmap_scaled.astype('uint8')

    def generateLabels(self, heatmap):
        labels = label(heatmap)

        return labels

    def annotate(self, image, bounding_boxes, heatmap, labels):
        draw_img = np.copy(image)

        if self.parameters.Annotation.bounding_box_individual:
            for box in bounding_boxes:
                color = box[3]
                cv2.rectangle(draw_img,
                              box[0],
                              box[1],
                              color,
                              6)

        if self.parameters.Annotation.bounding_box_heatmap_average:
            for car_number in range(1, labels[1] + 1):
                # Find pixels with each car_number label value
                nonzero = (labels[0] == car_number).nonzero()
                # Identify x and y values of those pixels
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                # Define a bounding box based on min/max x and y
                bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
                # Draw the box on the image
                cv2.rectangle(draw_img, bbox[0], bbox[1], (255, 0, 0), 10)

        if self.parameters.Annotation.heatmap:
            draw_img = cv2.addWeighted(draw_img, 0.5, heatmap, 1, 0)

        return draw_img

    def processImage(self, image):


        bounding_boxes = self.findBoundingBoxes(image)
        heatmap = self.generateHeatMap(image, bounding_boxes, self.parameters.SearchSettings.threshold)
        labels = self.generateLabels(heatmap)

        return self.annotate(image, bounding_boxes, heatmap, labels)

    def openVideo(self, file_name, start_time, end_time, frame_rate):
        num_Frames = (end_time - start_time) * frame_rate + 1

        clip1 = VideoFileClip(file_name).subclip(start_time, end_time)  # .subclip(21,24)#.subclip(40,43)

        return clip1


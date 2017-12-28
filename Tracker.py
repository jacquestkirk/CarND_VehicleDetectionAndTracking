from FeatureExtractor import *
from moviepy.editor import VideoFileClip

class Tracker:
    def __init__(self, featureList, parameters, svc, scaler ):
        self.featureList = featureList
        self.parameters = parameters
        self.svc = svc
        self.scaler = scaler

        self.featureExtractor = FeatureExtractor(self.parameters, self.featureList)



    def processImage(self, image):

        draw_img = np.copy(image)
        img = image.astype(np.float32) / 255

        img_tosearch = img[self.parameters.SearchSettings.y_start:self.parameters.SearchSettings.y_stop, :, :]
        ctrans_tosearch = FeatureExtractor.convert_color(img_tosearch, enum_ColorSpaces.ycrcb)
        if self.parameters.SearchSettings.scale != 1:
            imshape = ctrans_tosearch.shape
            img_converted = cv2.resize(ctrans_tosearch,
                                       (np.int(imshape[1] / self.parameters.SearchSettings.scale),
                                        np.int(imshape[0] / self.parameters.SearchSettings.scale)))


        # Define blocks and steps as above
        nxblocks = (img_converted.shape[1] // self.parameters.HogSettings.pix_per_cell) - self.parameters.HogSettings.cell_per_block + 1
        nyblocks = (img_converted.shape[0] // self.parameters.HogSettings.pix_per_cell) - self.parameters.HogSettings.cell_per_block + 1
        nfeat_per_block = self.parameters.HogSettings.orient * self.parameters.HogSettings.cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.parameters.HogSettings.pix_per_cell) - self.parameters.HogSettings.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
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
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = FeatureExtractor.bin_spatial(subimg, size=self.parameters.SpatialSettings.new_size)
                hist_features = FeatureExtractor.color_hist(subimg, nbins=self.parameters.HistogramSettings.bins)

                # Scale features and make a prediction
                test_features = self.scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * self.parameters.SearchSettings.scale)
                    ytop_draw = np.int(ytop * self.parameters.SearchSettings.scale)
                    win_draw = np.int(window * self.parameters.SearchSettings.scale)
                    cv2.rectangle(draw_img,
                                  (xbox_left, ytop_draw + self.parameters.SearchSettings.y_start),
                                  (xbox_left + win_draw, ytop_draw + win_draw + self.parameters.SearchSettings.y_start),
                                  (0, 0, 255),
                                  6)

        return draw_img

    def openVideo(self, file_name, start_time, end_time, frame_rate):
        num_Frames = (end_time - start_time) * frame_rate + 1

        clip1 = VideoFileClip(file_name)#.subclip(start_time, end_time)  # .subclip(21,24)#.subclip(40,43)

        return clip1


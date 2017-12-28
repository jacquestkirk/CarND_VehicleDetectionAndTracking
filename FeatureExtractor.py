import numpy as np
import cv2
from skimage.feature import hog

class enum_ColorSpaces:
    rgb = 0
    hsv = 1
    ycrcb = 2
    luv = 3
    hls = 4
    greyscale = 5
    none = 6


class enum_FeatureExtractor:
    hog = 0
    spatial = 1
    color = 2
    none = 3

class enum_FeatureExtractorType:
    train = 0
    predict = 1

class Feature:
    def __init__(self, colorSpace, featureExtractor):
        self.colorSpace = colorSpace
        self.featureExtractor = featureExtractor



class FeatureExtractor:
    def __init__(self, parameters, features_list, feature_extractor_type = enum_FeatureExtractorType.train):
        self.features_list = features_list
        self.image = None
        self.parameters = parameters
        self.hogFeatures = None
        self.feature_extractor_type = feature_extractor_type

    def preCalculateHogFeatures(self, image, orient, pix_per_cell, cell_per_block, feature_vec):
        hog_features = []

        for i in range(image.shape[2]):  # repeat for each color channel
            features = hog(image[:, :, i], orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           transform_sqrt=False,
                           visualise=False, feature_vector=feature_vec)
            hog_features.append(features)

        self.hogFeatures = hog_features

        return hog_features





    def findAllFeatures(self, image):
        self.image = image

        allFeatures = None

        for feature in self.features_list:
            extractedFeature = self.findFeature(image, feature)


            if allFeatures == None:
                allFeatures = extractedFeature
            else:
                allFeatures = np.concatenate([allFeatures, extractedFeature])

        return allFeatures


    def findFeature(self, image, feature):

        color_converted = FeatureExtractor.convert_color(image, feature.colorSpace)
        feature_extracted = self.extractFeature(color_converted, feature.featureExtractor, self.parameters)
        return feature_extracted


    def extractFeature(self, image, feature_extraction_type, all_parameters):

        if feature_extraction_type == enum_FeatureExtractor.hog:
            hogParameters = all_parameters.HogSettings
            if(self.feature_extractor_type == enum_FeatureExtractorType.predict):
                features = get_hog_from_precompute(image)
            else:
                features = FeatureExtractor.get_hog_features(image,
                                                  hogParameters.orient,
                                                  hogParameters.pix_per_cell,
                                                  hogParameters.cell_per_block,
                                                  hogParameters.feature_vec)

        elif feature_extraction_type == enum_FeatureExtractor.spatial:
            spatialParameters = all_parameters.SpatialSettings
            features = FeatureExtractor.bin_spatial(image,
                                         spatialParameters.new_size)

        elif feature_extraction_type == enum_FeatureExtractor.color:
            histParameters = all_parameters.HistogramSettings
            features = FeatureExtractor.color_hist(image, histParameters.bins)

        return features


    def get_hog_from_precompute(self):
        ypos = yb * cells_per_step
        xpos = xb * cells_per_step

        hog_feat1 = self.hogFeatures[0][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
        hog_feat2 = self.hogFeatures[0][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
        hog_feat3 = self.hogFeatures[0][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

        return hog_features

    @staticmethod
    def convert_color(image, color_space):

        if color_space == enum_ColorSpaces.rgb:
            image_converted = np.copy(image)
        elif color_space == enum_ColorSpaces.hsv:
            image_converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == enum_ColorSpaces.luv:
            image_converted = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == enum_ColorSpaces.hls:
            image_converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == enum_ColorSpaces.luv:
            image_converted = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == enum_ColorSpaces.ycrcb:
            image_converted = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        elif color_space == enum_ColorSpaces.greyscale:
            image_converted = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_converted = np.copy(image)

        return image_converted

    @staticmethod
    def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                         vis=False, feature_vec=True):
        hog_features = []

        for i in range(img.shape[2]): #repeat for each color channel
            features = hog(img[:, :, i], orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           transform_sqrt=False,
                           visualise=vis, feature_vector=feature_vec)
            hog_features.append(features)

        hog_features = np.ravel(hog_features)

        return hog_features

    @staticmethod
    def bin_spatial(img, size=(32, 32)):
        color1 = cv2.resize(img[:, :, 0], size).ravel()
        color2 = cv2.resize(img[:, :, 1], size).ravel()
        color3 = cv2.resize(img[:, :, 2], size).ravel()
        return np.hstack((color1, color2, color3))

    @staticmethod
    def color_hist(img, nbins=32):  # bins_range=(0, 256)
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features
from Parameters import *
from FeatureExtractor import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Trainer:
    def __init__(self, featureList, parameters):
        self.featureList = featureList
        self.parameters = parameters
        self.svc = None
        self.scaler = None

        self.featureExtractor = FeatureExtractor(self.parameters, self.featureList)

        self.carData = None
        self.notCarData = None
        self.carPath = ["training_images\\vehicles\\vehicles\\GTI_Far",
                        "training_images\\vehicles\\vehicles\\GTI_Left",
                        "training_images\\vehicles\\vehicles\\GTI_MiddleClose",
                        "training_images\\vehicles\\vehicles\\GTI_Right",
                        "training_images\\vehicles\\vehicles\\KITTI_extracted"]
        self.notCarPath = ["training_images\\non-vehicles\\non-vehicles\\Extras",
                        "training_images\\non-vehicles\\non-vehicles\\GTI",]

        self.carFeatures = None
        self.notCarFeatures = None


    def buildTrainingSet(self):
        #load car images

        self.carData = []
        self.notCarData = []

        #Car data
        for path in self.carPath:
            globPath = path + "\\*.png"
            images = glob.glob(globPath)
            for image in images:
                self.carData.append(image)

        #Non car data
        for path in self.notCarPath:
            globPath = path + "\\*.png"
            images = glob.glob(globPath)
            for image in images:
                self.notCarData.append(image)

    def truncateTrainingSet(self, cutoff):
        self.carData = self.carData[0:cutoff]
        self.notCarData = self.notCarData[0:cutoff]

    def train(self):

        #extract features
        t = time.time()
        self.carFeatures = self.extractFeatures(self.carData)
        self.notCarFeatures = self.extractFeatures(self.notCarData)

        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract features...')

        # Create an array stack of feature vectors
        X = np.vstack((self.carFeatures, self.notCarFeatures)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(self.carFeatures)), np.zeros(len(self.notCarFeatures))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

        self.svc = svc
        self.scaler = X_scaler

        return

    def extractFeatures(self, imageList):

        featuresList = []
        for imageFile in imageList:
            image = mpimg.imread(imageFile)
            features = self.featureExtractor.findAllFeatures(image)
            featuresList.append(features)

        return featuresList



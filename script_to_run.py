from Trainer import *
from Tracker import *
from FeatureExtractor import *
from Parameters import *
import pickle

train = False
featureList = []
featureList.append(Feature(enum_ColorSpaces.ycrcb, enum_FeatureExtractor.hog))
featureList.append(Feature(enum_ColorSpaces.ycrcb, enum_FeatureExtractor.spatial))
featureList.append(Feature(enum_ColorSpaces.ycrcb, enum_FeatureExtractor.color))


if (train):
    trainer = Trainer(featureList, Parameters)
    trainer.buildTrainingSet()
    #trainer.truncateTrainingSet(500)
    trainer.train()
    pickle.dump([trainer.svc, trainer.scaler], open("trainingData.p", "wb"))
    svc = trainer.svc
    scaler = trainer.scaler
else:
    [svc, scaler] = pickle.load(open("trainingData.p", "rb"))




file_name = 'test_video.mp4'

tracker = Tracker(featureList, Parameters, svc, scaler)
clip1 = tracker.openVideo(file_name, 0, 5, 25)


def process_image(image):
    result = tracker.processImage(image)
    return result


newClip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
newClip.write_videofile('processed.mp4', audio=False)

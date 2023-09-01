
# program to test whether a class is pickle-able or not

import sys
from pathlib import Path
group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

import pickle
from utils.ClassifierKerasTrinary import ClassifierKerasTrinary
from utils.CustomWeightedLoss import CustomWeightedLoss
import NNTClassifier

def pickle_obj(obj):
    pstr = ""

    try:
        pstr = pickle.dumps(obj)
    except pickle.PicklingError as e:
        print(f'An error occurred while pickling: {e}')


    try:
        obj_loaded = pickle.loads(pstr)
    except pickle.UnpicklingError as e:
        print(f'An error occurred while unpickling: {e}')


# create an instance, then just try to save and reload

classifier_type = NNTClassifier.ClassifierType.LSTM
clf = ClassifierKerasTrinary(pair="SOL/USDT", seq_len=4, num_features=16)
clf, name = NNTClassifier.create_classifier(classifier_type, pair="SOL/USDT", seq_len=4, nfeatures=16)

pickle_obj(clf)

loss = CustomWeightedLoss(CustomWeightedLoss.WeightedLossType.WEIGHTED_CATEGORICAL, [])
pickle_obj(loss)




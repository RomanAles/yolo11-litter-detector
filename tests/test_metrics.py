import pytest
from litter_detector.metrics import average_precision,count_at


def row(predictions,truth=None):
    return {'width':100,'height':100,'truth':truth if truth is not None else [[.1,.1,.3,.3]],'predictions':predictions}


def test_perfect_single_class_ap_and_recall():
    rows=[row([[.1,.1,.3,.3,.9]])]
    assert average_precision(rows)['map50_95']==pytest.approx(1)
    point=count_at(rows,.5);assert point['tp']==1 and point['fp']==0 and point['small_recall']==1


def test_duplicate_detection_does_not_match_twice():
    point=count_at([row([[.1,.1,.3,.3,.9],[.1,.1,.3,.3,.8]])],.5)
    assert point['tp']==1 and point['fp']==1 and point['fn']==0


def test_negative_scenes_and_misses_are_counted():
    rows=[row([],[[.1,.1,.3,.3]]),row([[.1,.1,.3,.3,.9]],[])]
    point=count_at(rows,.5);assert point['fn']==1 and point['negative_false_positives']==1 and point['recall']==0
    assert average_precision(rows)['ap50']==0

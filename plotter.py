from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import argparse
import utils

parser = argparse.ArgumentParser(description='Evaluation script')
parser.add_argument('--gt', type=str, default='PennFudanPed_val.json')
parser.add_argument('--pred_cv2', type=str, default='hog_cv2_output_file.json')
parser.add_argument('--pred_svm', type=str, default='scikit_svm_output_file.json')
parser.add_argument('--pred_frcnn', type=str, default='frcnn_output_file.json')


args = parser.parse_args()

anno = COCO(args.gt)  # init annotations api
pred_cv2 = anno.loadRes(args.pred_cv2)  # init predictions api
pred_svm = anno.loadRes(args.pred_svm)
pred_frcnn = anno.loadRes(args.pred_frcnn)
test_indices = range(len(anno.imgs))

eval_cv2 = COCOeval(anno, pred_cv2, 'bbox')
eval_svm = COCOeval(anno, pred_svm, 'bbox')
eval_frcnn = COCOeval(anno, pred_frcnn, 'bbox')
eval_cv2.params.imgIds = test_indices
eval_svm.params.imgIds = test_indices
eval_frcnn.params.imgIds = test_indices

eval_cv2.evaluate()
eval_cv2.accumulate()
eval_cv2.summarize()
eval_svm.evaluate()
eval_svm.accumulate()
eval_svm.summarize()
eval_frcnn.evaluate()
eval_frcnn.accumulate()
eval_frcnn.summarize()

# utils.plot_roc(0.9, eval_cv2.eval['precision'][8, :, 1, 3, 2], eval_svm.eval['precision'][8, :, 1, 3, 2], eval_frcnn.eval['precision'][8, :, 1, 3, 2])

cv2_pr = np.average(eval_cv2.eval['precision'][:, :, 1, 3, 2], axis=1)
cv2_re = eval_cv2.eval['recall'][:, 1, 3, 2]
svm_pr = np.average(eval_svm.eval['precision'][:, :, 1, 3, 2], axis=1)
svm_re = eval_svm.eval['recall'][:, 1, 3, 2]
frcnn_pr = np.average(eval_frcnn.eval['precision'][:, :, 1, 3, 2], axis=1)
frcnn_re = eval_frcnn.eval['recall'][:, 1, 3, 2]
utils.plot_PR_across_ious(cv2_pr, svm_pr, frcnn_pr, cv2_re, svm_re, frcnn_re)
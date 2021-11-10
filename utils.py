import matplotlib.pyplot as plt
import numpy as np

def plot_roc(iou_threshold, hog_cv2_values, hog_svm_values, frcnn_values):
    recall = np.linspace(0.0, 1.0, 101)
    plt.plot(recall, hog_cv2_values, label='hog_cv2')
    plt.plot(recall, hog_svm_values, label='hog_svm')
    plt.plot(recall, frcnn_values, label='frcnn')
    plt.legend()
    plt.savefig('PrRe_iou_th_{}.png'.format(iou_threshold), bbox_inches='tight')

def plot_PR_across_ious(hog_cv2_pr, hog_svm_pr, frcnn_pr, hog_cv2_re, hog_svm_re, frcnn_re):
    ious = np.linspace(0.5, 0.95, 10)
    plt.plot(hog_cv2_re, hog_cv2_pr, 'o-', label='hog_cv2')
    plt.plot(hog_svm_re, hog_svm_pr, 'o-', label='hog_svm')
    plt.plot(frcnn_re, frcnn_pr, 'o-', label='frcnn')
    plt.legend()
    plt.savefig('PrRe_across_ious.png', bbox_inches='tight')
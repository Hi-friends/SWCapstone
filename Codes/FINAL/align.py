import numpy as np
import cv2
import os

def align_trainset() :
    # load trainset landmark file & affine transformation
    path = 'landmark/'

    file_lst = os.listdir(path)
    file_list_npy = [f for f in file_lst if f.endswith(".npy")]
    for file in file_list_npy :
        TEMPLATE = np.load('{0}'.format(path+file),allow_pickle=True)
        TEMPLATE = TEMPLATE[0]
        TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
        MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
        OUTER_EYES_AND_NOSE = [36, 45, 33]
        npLandmarkIndices = np.array(OUTER_EYES_AND_NOSE)
        H = cv2.getAffineTransform(TEMPLATE[npLandmarkIndices], 96 * MINMAX_TEMPLATE[npLandmarkIndices])
        whole = []
        for i in TEMPLATE :
            lst = list(i)
            lst.append(1)
            whole.append(lst)
        landmark = np.array(whole)
        affine = []
        for i in landmark :
            val = H @ i
            affine.append(list(val))
        affine = np.array(affine)
        np.save("affine_landmark/affine_{0}".format(file),affine)

def align_testset() :
    # load testset landmark file & affine transformation
    path = 'landmark/test/'

    file_lst = os.listdir(path)
    file_list_npy = [f for f in file_lst if f.endswith(".npy")]
    for file in file_list_npy :
        TEMPLATE = np.load('{0}'.format(path+file),allow_pickle=True)
        TEMPLATE = TEMPLATE[0]
        TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
        MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
        OUTER_EYES_AND_NOSE = [36, 45, 33]
        npLandmarkIndices = np.array(OUTER_EYES_AND_NOSE)
        H = cv2.getAffineTransform(TEMPLATE[npLandmarkIndices], 96 * MINMAX_TEMPLATE[npLandmarkIndices])
        whole = []
        for i in TEMPLATE :
            lst = list(i)
            lst.append(1)
            whole.append(lst)
        landmark = np.array(whole)
        affine = []
        for i in landmark :
            val = H @ i
            affine.append(list(val))
        affine = np.array(affine)
        np.save("affine_landmark/test/affine_{0}".format(file),affine)

def trainset_get_result() :
    file_lst = os.listdir('./landmark/')
    file_list_npy = [f for f in file_lst if f.endswith(".npy")]
    for file in file_list_npy:
        TEMPLATE = np.load('./landmark/{0}'.format(file),allow_pickle=True)
        TEMPLATE = TEMPLATE[0]
        TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
        MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
        OUTER_EYES_AND_NOSE = [36, 45, 33]
        npLandmarkIndices = np.array(OUTER_EYES_AND_NOSE)
        imgDim=96
        rgbImg = cv2.imread('./image/{0}.jpg'.format(file.split('.')[0]))
        H = cv2.getAffineTransform(TEMPLATE[npLandmarkIndices], imgDim * MINMAX_TEMPLATE[npLandmarkIndices])
        thumbnail = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))
        #cv2.imshow('Image', thumbnail)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite('./transformed/{0}_test_result.jpg'.format(file.split('.')[0]), thumbnail)

def testset_get_result() :
    file_lst = os.listdir('./landmark/test/')
    file_list_npy = [f for f in file_lst if f.endswith(".npy")]
    for file in file_list_npy:
        TEMPLATE = np.load('./landmark/test/{0}'.format(file),allow_pickle=True)
        TEMPLATE = TEMPLATE[0]
        TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
        MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
        OUTER_EYES_AND_NOSE = [36, 45, 33]
        npLandmarkIndices = np.array(OUTER_EYES_AND_NOSE)
        imgDim=96
        rgbImg = cv2.imread('./image/test/{0}.jpg'.format(file.split('.')[0]))
        H = cv2.getAffineTransform(TEMPLATE[npLandmarkIndices], imgDim * MINMAX_TEMPLATE[npLandmarkIndices])
        thumbnail = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))
        #cv2.imshow('Image', thumbnail)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite('./transformed/test/{0}_test_result.jpg'.format(file.split('.')[0]), thumbnail)

print('Choose dataset you want to transform')
dataset = input('trainset  / testset / both : ')
not_choosed = True

while not_choosed :
    if dataset == 'trainset' :
        align_trainset()
        trainset_get_result()
        not_choosed = False
    elif dataset == 'testset' :
        align_testset()
        testset_get_result()
        not_choosed  = False
    elif dataset == 'both' :
        align_trainset()
        trainset_get_result()
        align_testset()
        testset_get_result()
        not_choosed = False
    else :
        print('type again')
        dataset = input('trainset  / testset / both : ')
    

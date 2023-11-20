import os

import cv2
import numpy as np
import imutils
import lip
import dlib
from imutils import face_utils

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_hsv_mask(img, debug=False):
    assert isinstance(img, np.ndarray), 'image must be a np array'
    assert img.ndim == 3, 'skin detection can only work on color images'

    lower_thresh = np.array([0, 50, 0], dtype=np.uint8)
    upper_thresh = np.array([120, 150, 255], dtype=np.uint8)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    msk_hsv = cv2.inRange(img_hsv, lower_thresh, upper_thresh)

    #msk_hsv[msk_hsv < 128] = 0
    #msk_hsv[msk_hsv >= 128] = 1

    return msk_hsv.astype(float)


def get_rgb_mask(img, debug=False):
    assert isinstance(img, np.ndarray), 'image must be a np array'
    assert img.ndim == 3, 'skin detection can only work on color images'

    lower_thresh = np.array([45, 52, 108], dtype=np.uint8)
    upper_thresh = np.array([255, 255, 255], dtype=np.uint8)

    mask_a = cv2.inRange(img, lower_thresh, upper_thresh)
    mask_b = 255 * ((img[:, :, 2] - img[:, :, 1]) / 20)
    mask_c = 255 * ((np.max(img, axis=2) - np.min(img, axis=2)) / 20)

    mask_a = mask_a.astype(float)

    msk_rgb = cv2.bitwise_and(mask_a, mask_b)
    msk_rgb = cv2.bitwise_and(mask_c, msk_rgb)

    msk_rgb[msk_rgb < 128] = 0
    msk_rgb[msk_rgb >= 128] = 255

    return msk_rgb.astype(float)


def get_ycrcb_mask(img, debug=False):
    assert isinstance(img, np.ndarray), 'image must be a np array'
    assert img.ndim == 3, 'skin detection can only work on color images'

    lower_thresh = np.array([90, 100, 130], dtype=np.uint8)#90
    upper_thresh = np.array([230, 120, 180], dtype=np.uint8)#230

    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    msk_ycrcb = cv2.inRange(img_ycrcb, lower_thresh, upper_thresh)

    #msk_ycrcb[msk_ycrcb < 128] = 0
    #msk_ycrcb[msk_ycrcb >= 128] = 1

    return msk_ycrcb.astype(float)


def grab_cut_mask(img_col, mask, debug=False):
    assert isinstance(img_col, np.ndarray), 'image must be a np array'
    assert isinstance(mask, np.ndarray), 'mask must be a np array'
    assert img_col.ndim == 3, 'skin detection can only work on color images'
    assert mask.ndim == 2, 'mask must be 2D'

    kernel = np.ones((50, 50), np.float32) / (50 * 50)
    dst = cv2.filter2D(mask, -1, kernel)
    dst[dst != 0] = 255
    free = np.array(cv2.bitwise_not(dst), dtype=np.uint8)

    grab_mask = np.zeros(mask.shape, dtype=np.uint8)
    grab_mask[:, :] = 2
    grab_mask[mask == 255] = 1
    grab_mask[free == 255] = 0

    if np.unique(grab_mask).tolist() == [0, 1]:
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        if img_col.size != 0:
            mask, bgdModel, fgdModel = cv2.grabCut(img_col, grab_mask, None, bgdModel, fgdModel, 5,
                                                   cv2.GC_INIT_WITH_MASK)
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        else:
            print('img_col is empty')

    return mask


def closing(mask):
    assert isinstance(mask, np.ndarray), 'mask must be a np array'
    assert mask.ndim == 2, 'mask must be a greyscale image'

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask


def process(img, thresh=0.5, debug=False):
    assert isinstance(img, np.ndarray), 'image must be a np array'
    assert img.ndim == 3, 'skin detection can only work on color images'

    mask_hsv = get_hsv_mask(img, debug=debug)
    mask_rgb = get_rgb_mask(img, debug=debug)
    mask_ycrcb = get_ycrcb_mask(img, debug=debug)

    n_masks = 3.0
    mask = np.where((mask_hsv + mask_rgb + mask_ycrcb)/3>255/2,255,0)

    #mask[mask < thresh] = 0.0
    #mask[mask >= thresh] = 255.0)

    mask = mask.astype(np.uint8)
    mask = closing(mask)
    mask = grab_cut_mask(img, mask, debug=debug)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=11)
    return mask

def draw_delaunay(img, subdiv, delaunay_color) :
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in triangleList :
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

gmin = -1
def findtop(img, coord):
    global gmin
    mask = process(img)
    x, y = coord
    y = y-10
    if(gmin == -1):
        gmin = y+1
    else:
        if(gmin+20<y):
            y = gmin+20
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    while(1):
        if(y==0):
            break
        y-=1
        if(abs(l.item(y,x)-l.item(y+1,x))>=15 or mask.item(y,x)==0):
            break
    gmin = y

    return (x,y)


def triangulate(image):
    global gmin

    # load the input image, resize it, and convert it to grayscale
    image = imutils.resize(image, width=500)
    img = image.copy()  # used in manual editing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a np array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cx = int(.15*w)
        cy = int(.5*h)

        # show the face number
        subdiv = cv2.Subdiv2D((max(x-cx,0), max(y-cy,0), min(w+x+cx,image.shape[1]), min(h+y+cx,image.shape[0])))
        forehead = []
        gmin = -1
        for num, (x, y) in enumerate(shape):


            if((num>=17 and num<=27)):
                forehead.append(findtop(image, (x,y)))

            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        for item in forehead:
            shape = np.vstack((shape, item))

        for (x, y) in shape:
            subdiv.insert((int(x),int(y)))

        return shape, subdiv.getTriangleList()

def warp(src, dst):
    src_points, src_triangles = triangulate(src)
    dst_points, dst_triangles = triangulate(dst)
    warped_image = np.zeros(src.shape, dtype=np.uint8)

    for i,t in enumerate(dst_triangles):
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        first, second, third = -1,-1,-1
        for i2, (x,y) in enumerate(dst_points):
            if(x==t[0] and y==t[1]):
                first = i2
            if(x==t[2] and y==t[3]):
                second = i2
            if(x==t[4] and y==t[5]):
                third = i2
        if(first>=0 and second>=0 and third>=0):
            x1,y1 = src_points[first]
            x2,y2 = src_points[second]
            x3,y3 = src_points[third]
            dx1,dy1 = dst_points[first]
            dx2,dy2 = dst_points[second]
            dx3,dy3 = dst_points[third]

            #creating mask in destination image
            mask = np.zeros(src.shape, dtype=np.uint8)
            roi_corners = np.array([[dx1,dy1],[dx2,dy2],[dx3,dy3]], dtype=np.int32)
            cv2.fillPoly(mask, [roi_corners], (255,255,255))

            #warping src image to destination image
            pts1 = np.float32([[x1,y1],[x2,y2],[x3,y3]])
            pts2 = np.float32([[dx1,dy1],[dx2,dy2],[dx3,dy3]])
            M = cv2.getAffineTransform(pts1,pts2)
            rows,cols,ch = src.shape
            res = cv2.warpAffine(src,M,(cols,rows))

            warped_image = cv2.bitwise_or(warped_image,cv2.bitwise_and(mask,res))

            '''
            cv2.line(src, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA, 0)
            cv2.line(src, pt2, pt3, (255, 255, 255), 1, cv2.LINE_AA, 0)
            cv2.line(src, pt3, pt1, (255, 255, 255), 1, cv2.LINE_AA, 0)
            '''

    return warped_image

def find_mask(image, betamap):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a np array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the face parts individually
        mask = np.zeros(image.shape, dtype=image.dtype)
        noseMask = np.zeros(image.shape, dtype=image.dtype)
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)

            if(betamap):
                if(name=='right_eyebrow' or name=='left_eyebrow'):
                    continue
                if(name=='jaw'):
                    continue
            else:
                if(name=='jaw' or name=='nose' or name=='left_eyebrow' or name=='right_eyebrow'):
                    continue
            pts = shape[i:j]
            hull = cv2.convexHull(pts)
            if(name=='nose'):
                cv2.drawContours(noseMask, [hull], -1, (255,255,255), -1)
            else:
                cv2.drawContours(mask, [hull], -1, (255,255,255), -1)

        if(betamap):
            kernel = np.ones((5,5),np.uint8)
            dilation = cv2.dilate(mask,kernel,iterations = 4)
            gradient = cv2.morphologyEx(noseMask, cv2.MORPH_GRADIENT, kernel)
            gradient = cv2.dilate(gradient,kernel,iterations = 2)
            mask = dilation+gradient

        return mask

def overlay(orig, makeup, mask):

    blur_mask = cv2.blur(mask, (20, 20))
    new = makeup.copy()
    for y in range(0, orig.shape[0]):
        for x in range(0, orig.shape[1]):
            w = blur_mask[y][x]/255
            if (w > 0.6):
                w = (w - 0.6) / 0.4
            else:
                w = 0
            new[y][x] = makeup[y][x]*w + orig[y][x]*(1 - w)

    return new

def decompose(img):
    base = cv2.bilateralFilter(img, 9, 75,75)
    return base, img-base

def warp_target(subject, target):

    if(target.shape[0]>subject.shape[0]):
        new_subject = np.zeros((target.shape[0]-subject.shape[0],subject.shape[1],3), dtype=subject.dtype)
        subject = np.vstack((subject, new_subject))
    else:
        #resizing target
        new_target = np.zeros((subject.shape[0]-target.shape[0],target.shape[1],3), dtype=target.dtype)
        target = np.vstack((target, new_target))

    if(subject.shape[0]%2!=0):
        zero_layer = np.zeros((1, target.shape[1],3), dtype=target.dtype)
        target = np.vstack((target, zero_layer))
        subject = np.vstack((subject, zero_layer))

    warped_target = warp(target, subject)

    return subject, warped_target

def apply_makeup(subject, warped_target):
    zeros = np.zeros(warped_target.shape, dtype=warped_target.dtype)
    ones = np.ones(warped_target.shape, dtype=warped_target.dtype)
    face_mask = np.where(warped_target==[0,0,0], zeros, ones*255)
    # cv2.imshow('mask', face_mask)
    # cv2.waitKey(0)
    
    sub_lab = cv2.cvtColor(subject, cv2.COLOR_BGR2LAB)
    tar_lab = cv2.cvtColor(warped_target, cv2.COLOR_BGR2LAB)

    sl, sa, sb = cv2.split(sub_lab)
    tl, ta, tb = cv2.split(tar_lab)

    face_struct_s, skin_detail_s = decompose(sl)
    face_struct_t, skin_detail_t = decompose(tl)

    #color transfer
    gamma = .8
    '''
    type = sa.dtype
    sa.dtype = float
    ta.dtype = float
    sb.dtype = float
    tb.dtype = float
    '''
    type = sa.dtype
    ra = np.where(True, sa*(1-gamma)+ta*gamma, zeros[:,:,0])
    rb = np.where(True, sb*(1-gamma)+tb*gamma, zeros[:,:,0])
    ra = ra.astype(type)
    rb = rb.astype(type)
    ra = cv2.bitwise_and(ra,ra,mask = face_mask[:,:,0])
    rb = cv2.bitwise_and(rb,rb,mask = face_mask[:,:,0])

    gammaI = 0
    gammaE = 1
    skin_detail_r = np.where(True, skin_detail_s*gammaI + skin_detail_t*gammaE, zeros[:,:,0])
    skin_detail_r = skin_detail_r.astype(type)

    fp_mask = find_mask(subject, True)
    src_gauss = cv2.pyrDown(face_struct_s)
    src_lapla = face_struct_s - cv2.pyrUp(src_gauss)
    dst_gauss = cv2.pyrDown(face_struct_t)
    dst_lapla = face_struct_t - cv2.pyrUp(dst_gauss)
    face_struct_r = np.where(face_mask[:,:,0]==0, face_struct_s, dst_lapla + cv2.pyrUp(src_gauss))

    face_struct_r = np.where(fp_mask[:,:,0]==255, face_struct_s, face_struct_r)

    rl = face_struct_r+skin_detail_r
    rl = cv2.bitwise_and(rl,rl,mask = face_mask[:,:,0])

    res_lab = cv2.merge((rl, ra, rb))
    res = cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)

    fp_mask = find_mask(subject, False)
    res = cv2.bitwise_and(res,res,mask = face_mask[:,:,0])
    res = np.where(face_mask==[0,0,0], subject, res)
    res = np.where(fp_mask==[255,255,255], subject, res)


    #apply lip makeup
    M, lip_map = lip.lip_makeup(subject, warped_target)
    res = np.where(lip_map==[255,255,255], M, res)
    res = overlay(subject, res, face_mask[:,:,0])
    return res

def makeup_main(url_subject, url_target):
    folder = os.path.join(os.path.dirname('/Users/hatk/Desktop/FaceMakeUpProject/'))
    subject = cv2.imread(folder + '/' + url_subject)
    target = cv2.imread(folder + '/' + url_target)
    subject = imutils.resize(subject, width=500)
    target = imutils.resize(target, width=500)
    sub, warped_tar = warp_target(subject, target)
    res = apply_makeup(sub, warped_tar)
    return res
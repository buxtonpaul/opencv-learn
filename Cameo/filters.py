''' filters for processing images'''
import cv2
import numpy
import utils

def recolorRC(src, dst):
    ''' Simulate conversion from BGR to Red/Cyan).
    Both source and dest are in BGR format
    dstb. = dst.g = 0.5 * (src.b + src.g)
    dst.r = src.r'''

    b, g, r = cv2.split(src)
    cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
    cv2.merge((b, b, r), dst)

def recolorRGV(src, dst):
    ''' Simulate conversion from BGR to RGV (red, green, value)
    Blues are desaturated
    dst.b = min (src.b, src,g, src.r)
    dst.g = src.g
    dst.r = src.r
    '''
    b, g, r = cv2.split(src)
    cv2.min(b, g, b)
    cv2.min(b, r, b)
    cv2.merge((b, g, r), dst)

def recolorCMV(src, dst):
    ''' something'''
    b, g, r = cv2.split(src)
    cv2.max(r, b, b)
    cv2.max(b, g, b)
    cv2.merge((b, g, r), dst)

class VFuncFilter(object):
    ''' Filter that applies function to V (or all of BGR)'''

    def __init__(self, vFunc=None, dtype=numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._vLookupArray = utils.createLookupArray(vFunc, length)

    def apply(self, src, dst):
        ''' Apply the filter with BGR or gray source/dest'''
        srcFlatView = utils.createFlatView(src)
        dstFlatView = utils.createFlatView(dst)
        utils.applyLookupArray(self._vLookupArray, srcFlatView, dstFlatView)

class VCurveFilter(VFuncFilter):
    '''A filter that applies a curve to V or all of BGR'''
    def __init__(self, vPoints, dtype=numpy.uint8):
        VFuncFilter.__init__(self, utils.createCurveFunc(vPoints), dtype)

class BGRFuncFilter(object):
    ''' A filter class for applying curves to seperate BGR channels'''
    def __init__(self, vFunc=None, bFunc=None, gFunc=None, rFunc=None, dtype=numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._bLookupArray = utils.createLookupArray(utils.createCompositeFunc(bFunc, vFunc),
                                                     length)
        self._gLookupArray = utils.createLookupArray(utils.createCompositeFunc(gFunc, vFunc),
                                                     length)
        self._rLookupArray = utils.createLookupArray(utils.createCompositeFunc(rFunc, vFunc),
                                                     length)

    def apply(self, src, dst):
        ''' Apply the filter to seperate BGR channels'''
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)

class BGRCurveFilter(BGRFuncFilter):
    ''' A filter that applies different curves to each BGR channel'''
    def __init__(self, vPoints=None, bPoints=None, gPoints=None, rPoints=None, dtype=numpy.uint8):
        BGRFuncFilter.__init__(self, utils.createCurveFunc(vPoints),
                               utils.createCurveFunc(bPoints),
                               utils.createCurveFunc(gPoints),
                               utils.createCurveFunc(rPoints), dtype)

class BGRPortraCurveFilter(BGRCurveFilter):
    '''A filter that applies a Portra Curve the the BGR channels'''
    def __init__(self, dtype=numpy.uint8):

        BGRCurveFilter.__init__(self,
                                vPoints=[(0, 0), (23, 20), (157, 173), (255, 255)],
                                bPoints=[(0, 0), (41, 46), (231, 238), (255, 255)],
                                gPoints=[(0, 0), (52, 47), (189, 196), (255, 255)],
                                rPoints=[(0, 0), (69, 69), (213, 218), (255, 255)],
                                dtype=dtype)

class BGRProviaCurveFilter(BGRCurveFilter):
    '''A filter that applies a Portra Curve the the BGR channels'''
    def __init__(self, dtype=numpy.uint8):

        BGRCurveFilter.__init__(self,
                                bPoints=[(0, 0), (35, 35), (205, 227), (255, 255)],
                                gPoints=[(0, 0), (27, 21), (196, 207), (255, 255)],
                                rPoints=[(0, 0), (59, 54), (202, 210), (255, 255)],
                                dtype=dtype)

class BGRVelviaCurveFilter(BGRCurveFilter):
    '''A filter that applies a Portra Curve the the BGR channels'''
    def __init__(self, dtype=numpy.uint8):

        BGRCurveFilter.__init__(self,
                                vPoints=[(0, 0), (128, 118), (221, 215), (255, 255)],
                                bPoints=[(0, 0), (25, 21), (122, 153), (255, 255)],
                                gPoints=[(0, 0), (25, 21), (95, 102), (255, 255)],
                                rPoints=[(0, 0), (41, 28), (183, 209), (255, 255)],
                                dtype=dtype)

class BGRCrossProcessCurveFilter(BGRCurveFilter):
    '''A filter that applies a Portra Curve the the BGR channels'''
    def __init__(self, dtype=numpy.uint8):

        BGRCurveFilter.__init__(self,
                                vPoints=[(0,0),(255,255)],
                                bPoints=[(0, 20), (255, 235)],
                                gPoints=[(0, 0), (56, 39), (208, 226), (255, 255)],
                                rPoints=[(0, 0), (56, 22), (211, 255), (255, 255)],
                                dtype=dtype)

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [numpy.hsplit(row, w//sx) for row in numpy.vsplit(img, h//sy)]
    cells = numpy.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells


def deSkew(src,dst):
    ''' Implementation of deskew using image moments, based on leanopencv.com example'''
    SZ=20
    graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    frm = split2d(graySrc,(SZ,SZ))
    m = cv2.moments(frm)
    if abs(m['mu02']) < 1e-2:
        # do deskewing needed.
        dst = frm
    # calculate skew based on central moments
    skew = m ['mu11']/m['mu02']
    # calculate affine transform to correct skewness
    M= numpy.float32([[1,skew, -0.5 * SZ * skew],[0,1,0]])
    # Apply affine trasform
    dst = cv2.warpAffine(frm,M, (SZ,SZ), flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)



def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    '''Function to apply a stroke edges filter to the frame'''
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0/255) * (255-graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)

class VConvolutionFilter(object):
    '''A filter that applies a convolution to V or all of BGR'''
    def __init__(self, kernel):
        self._kernel = kernel
    def apply(self, src, dst):
        '''Apply the filter kernel to the image'''
        cv2.filter2D(src, -1, self._kernel, dst)

class sharpenFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)

class findEdgesFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)

class blurFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)

class embossFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [0, 1, 2]])
        VConvolutionFilter.__init__(self,kernel)



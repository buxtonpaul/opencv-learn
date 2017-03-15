import cv2
import numpy
import scipy.interpolate

def createCurveFunc(points):
    ''' Interpolate a curve from the given control points'''
    if points is None:
        return None
    numPoints = len(points)
    if numPoints < 2:
        return None

    xs, ys = zip(*points)
    if numPoints < 4:
        kind = 'linear'
    else:
        kind = 'cubic'
    return scipy.interpolate.interp1d(xs, ys, kind, bounds_error=False)

def createLookupArray(func, length=256):
    ''' Return a lookup for a whole number input function'''

    if func is None:
        return None
    lookupArray = numpy.empty(length)
    i = 0
    while i < length:
        func_i = func(i)
        lookupArray[i] = min(max(0, func_i), length -1)
        i += 1
    return lookupArray

def applyLookupArray(lookupArray, src, dst):
    ''' Map a source to a destination using a lookup'''
    if lookupArray is None:
        return
    dst[:] = lookupArray[src]

def createCompositeFunc(func0, func1):
    ''' Return a composite of two functions'''
    if func0 is None:
        return
    if func1 is None:
        return
    return lambda x: func0(func1(x))

def createFlatView(array):
    ''' Return a 1d view of an array of any dimensionality'''
    flatView = array.view()
    flatView.shape = array.size
    return flatView

def isGray(image):
    ''' Return true if the image has one channel per pixel'''
    return image.ndim <3

def widthHeightDividedBy(image, divisor):
    ''' Return an images dimensions divided by a value'''
    h, w = image.shape[:2]
    return (w/divisor, h/divisor)


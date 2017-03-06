''' Managers module, contains capture manager and windowmanager classes'''
import time
import cv2
import numpy


class CaptureManager(object):
    ''' Capture manager class'''
    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):

        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview
        self._capture = capture
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

        self._startTime = None
        self._framesElapsed = long(0)
        self._fpsEstimate = None

    @property
    def frame(self):
        ''' returns the current frame'''
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
            return self._frame
    @property
    def isWritingImage(self):
        ''' Do we intent to write an image or not'''
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        '''Are we writing video or not'''
        return self._videoFilename is not None

    def enterFrame(self):
        '''Capture the next frame, if any'''

        # check that any previos frame was exited
        assert not self._enteredFrame, 'Previous frame not exited! (enter frame without exit frame'

        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        ''' Exit a frame if we have entered one!
        Draw to files, draw to window etc'''

        if self._frame is None:
            self._enteredFrame = False
            return
        # update FPS and related vars
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time()-self._startTime
            self._fpsEstimate = self._framesElapsed/ timeElapsed
        self._framesElapsed += 1

        # fraw to the window if present
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = numpy.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)

        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None

        if self.isWritingVideo:
            self._writeVideoFrame()

        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        ''' Write the next exited frame to an imagefile'''
        self._imageFilename = filename

    def startWritingVideo(self, filename, encoding=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        '''Start writing exited frames to a video file'''
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        ''' Stop writing video frames to a video file'''
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writeVideoFrame(self):
        if not self.isWritingVideo:
            return
        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                # Captures FPS is unknown so guess
                if self._framesElapsed < 20:
                    # wait until more frames have beeen captured before estimating
                    return
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(self._videoFilename, self._videoEncoding, fps, size)
        self._videoWriter.write(self._frame)


class WindowManager(object):
    ''' Window manager class'''
    def __init__(self, windowName, keypressCallback=None):
        self.keypressCallBack = keypressCallback

        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        ''' Return true if we have a created window'''
        return self._isWindowCreated

    def createWindow(self):
        ''' Create a window'''
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        ''' Show the current window'''
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        ''' Destroy the Current window'''
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        ''' Event handler code, waits for keys and calls the callback handler'''
        keycode = cv2.waitKey(1)
        if self.keypressCallBack is not None and keycode != -1:
            #Discard any non ascii info
            keycode &= 0xff
            self.keypressCallBack(keycode)

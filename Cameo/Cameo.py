''' Main application for the Cameo vision framework'''
import cv2
from managers import WindowManager, CaptureManager
import filters
import rects
from trackers import FaceTracker

class Cameo(object):
    ''' Cameo object for the vision framework'''
    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)

        self._curves = [None, filters.BGRCrossProcessCurveFilter(), filters.BGRPortraCurveFilter(),
                        filters.BGRProviaCurveFilter(), filters.BGRVelviaCurveFilter()]
        self._curveIndex = 0
        self._curveFilter = self._curves[self._curveIndex]

        self._recolorFilters = [None, filters.recolorCMV, filters.recolorRC, filters.recolorRGV]
        self._recolorIndex = 0
        self._recolor = self._recolorFilters[self._recolorIndex]

        self._convolutionFilters = [None, filters.findEdgesFilter(),
                                    filters.sharpenFilter(), filters.blurFilter(),
                                    filters.embossFilter()]
        self._convolutionIndex = 0
        self._convolution = self._convolutionFilters[self._convolutionIndex]

        self._strokeEdges = False

        self._faceTracker = FaceTracker()
        self._shouldDrawDebugRects = True

    def run(self):
        ''' Run the main loop'''

        self._windowManager.createWindow()
        self._windowManager.setStatus("K={},C={},R={},S={}".format(self._convolutionIndex,self._curveIndex,self._recolorIndex,self._strokeEdges))
        print"Cameo Vision Framework\n"\
             "Tab to start/stop recording\n"\
             "Space to grab a screenshot\n"\
             "r to cycle through recolor filters <none>, CMV, RC, RGV\n"\
             "c to cycle through tonemapping curves <none>,crossprocess, porta, provia, velvia\n"\
             "k to cycle through convolution filters <none>, find edges,sharpen, blur, emboss\n"\
             "s to apply stroke edges filter\n"
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            self._faceTracker.update(frame)
            faces = self._faceTracker.faces

            rects.swapRects(frame,frame,[face.faceRect for face in faces])

            if self._convolution is not None:
                self._convolution.apply(frame, frame)
            if self._curveFilter is not None:
                self._curveFilter.apply(frame, frame)
            if self._recolor is not None:
                self._recolor(frame, frame)
            if self._strokeEdges:
                filters.strokeEdges(frame, frame)
            
            filters.deSkew(frame,frame)

            if self._shouldDrawDebugRects:
                self._faceTracker.drawDebugRects(frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        ''' Handle keypresses
        Space -> take screenshot
        tab -> Start stop recording
        excape -> quit
        '''
        
        rawkey = keycode # for handling esc etc
        keycode = chr(keycode & 255)
        if rawkey == 0x20: # space
            self._captureManager.writeImage('screenshot.png')

        elif rawkey == 0x09: #tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif rawkey == 0x1b:  # escape
            self._windowManager.destroyWindow()
        
        elif keycode  in ['c','C']:
            self._curveIndex += 1
            if self._curveIndex >= len(self._curves):
                self._curveIndex = 0
            self._curveFilter = self._curves[self._curveIndex]

        elif keycode in ['r','R']:
            self._recolorIndex += 1
            if self._recolorIndex >= len(self._recolorFilters):
                self._recolorIndex = 0
            self._recolor = self._recolorFilters[self._recolorIndex]

        elif keycode in ['k','K']:
            self._convolutionIndex += 1
            if self._convolutionIndex >= len(self._convolutionFilters):
                self._convolutionIndex = 0
            self._convolution = self._convolutionFilters[self._convolutionIndex]
        elif keycode in ['s','S']:
            if self._strokeEdges:
                self._strokeEdges = False
            else:
                self._strokeEdges = True
        elif keycode in ['x','X']:
            if self._shouldDrawDebugRects:
                self._shouldDrawDebugRects = False
            else:
                self._shouldDrawDebugRects = True

        statusString="K={},C={},R={},f={}".format(self._convolutionIndex,self._curveIndex,self._recolorIndex,self._strokeEdges)
        self._windowManager.setStatus(statusString)


if __name__ == "__main__":
    Cameo().run()


    
import os
import cv2

class CameraUtils:
    def __init__(self, width=0, height=0, fps=30) -> None:
        # self._cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self._cam = cv2.VideoCapture(0)
        if not self._cam.isOpened():
            raise RuntimeError("Could not open camera.")
        self._cam.set(cv2.CAP_PROP_FPS, fps)
        self.frame_width = self._cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self._cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.saved_width = width if width != 0 else int(self._cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.saved_height = width if height != 0 else int(self._cam.get(cv2.CAP_PROP_FRAME_HEIGHT))


    def close(self):
        self._cam.release()
        cv2.destroyAllWindows()

    def read(self):
        ret, frame = self._cam.read()
        return frame
    
    def writeOutput(self, file_path, frame):
        folder = file_path.replace(file_path.split("/")[-1], '')
        if not os.path.exists(folder):
            os.mkdir(folder)
        cv2.imwrite(file_path, frame)
        print(f'{file_path} saved')

    def show(self, frame=None, title='Camera'):
        shown_frame = frame if frame is not None else self.read()
        cv2.imshow(title, cv2.flip(shown_frame, 1))
        return cv2.waitKey(1)
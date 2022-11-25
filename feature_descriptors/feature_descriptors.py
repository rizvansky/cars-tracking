import cv2
import numpy as np
from scipy.signal import convolve2d

from typing import Tuple

class BaseFeatureDescriptor:
    
    def __init__(self, ):
        pass
    
    def predict(self, image: np.ndarray):
        raise NotImplementedError('abstract method should be implemented')

    def _preprocess_image(
        self,
        raw_img: np.ndarray,
        target_size: Tuple[int, int] = None,
        target_colormap: int = None,
    ) -> np.ndarray:
        image = raw_img.copy()
        if target_size:
            image = cv2.resize(image, target_size)
        
        if target_colormap:
            image = cv2.cvtColor(image, target_colormap)
        
        return image
        
class HogFeatureDescriptor(BaseFeatureDescriptor):
    
    def __init__(self, group_size=3, step=2, bin_num=16) -> None:
        super().__init__()
        
        self.group_size = group_size
        self.step = group_size if step is None else step
        self.bin_num = bin_num
        
        self.default_size = (64, 64)
    
    def predict(self, image: np.ndarray, process_size=None):
        
        target_size = self.default_size if process_size is None else process_size
        
        img = self._preprocess_image(
            raw_img=image,
            target_size=target_size
        )
        
        mag, ang = self._compute_sobel(img)
        
        mag_cells, bin_cells = self._fill_cells(target_size, mag, ang, self.step)
        
        hists = self._compute_histogram(bin_cells, mag_cells)
        
        return np.hstack(hists)
        
    
    def _compute_sobel(self, img):
        gx = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
        gy = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=1)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        return mag, ang
    
    def _fill_cells(self, img_size, mag, ang, step):
        bins = np.int32(self.bin_num * ang / (2 * np.pi))
        
        h, w = img_size
        
        mag_cells = [
            mag[i : i + self.group_size, j : j + self.group_size]
            for i in range(0, h - self.group_size, step)
            for j in range(0, w - self.group_size, step)
        ]
        bin_cells = [
            bins[i : i + self.group_size, j : j + self.group_size]
            for i in range(0, h - self.group_size, step)
            for j in range(0, w - self.group_size, step)
        ]
        
        return mag_cells, bin_cells
        
    def _compute_histogram(self, bin_cells, mag_cells):
        hists = [
            np.bincount(b.ravel(), m.ravel(), self.bin_num) for b, m in zip(bin_cells, mag_cells)
        ]
        
        return hists
        

class FASTFeatureDescriptor(BaseFeatureDescriptor):
    
    def __init__(self, threshold=0.15, nms_window=2, N=9):
        super().__init__()
        
        self.threshold = threshold
        self.nms_window = nms_window
        
        self.kernel = np.array(
            [
                [1,2,1],
                [2,4,2],
                [1,2,1],
            ], dtype=np.float
        )
        self.kernel /= 16.0
        self.N = N
        
        self.cross_idx = np.array(
            [
                [3, 0, -3, 0],
                [0, 3, 0, -3],
            ],
        )
        self.circle_idx = np.array(
            [
                [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3],
                [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1],
            ],
        )
        
    def predict(self, image: np.array):
        image = self._preprocess_image(image, target_size=(64, 64), target_colormap=cv2.COLOR_BGR2GRAY)
        img = convolve2d(image, self.kernel, mode='same')
        
        keypoints, corner_img = self._detect_keypoints(img)
        fined_keypoints = self._apply_nms(keypoints, corner_img)
        
        return np.array(fined_keypoints)
        

    def _detect_keypoints(self, img):
        corner_img = np.zeros_like(img)
        keypoints = []
        for y in range(3, img.shape[0]-3):
            for x in range(3, img.shape[1]-3):
                Ip = img[y, x]
                t = self.threshold * Ip if self.threshold < 1 else self.threshold
                # fast checking cross idx only
                if np.count_nonzero(
                    Ip + t < img[
                        y + self.cross_idx[0, :],
                        x + self.cross_idx[1, :]
                    ]
                ) >= 3 or np.count_nonzero(
                        Ip - t > img[
                            y + self.cross_idx[0, :],
                            x + self.cross_idx[1, :]
                        ]
                    ) >= 3:
                    # detailed check -> full circle
                    if np.count_nonzero(
                        img[
                            y + self.circle_idx[0, :], 
                            x + self.circle_idx[1, :]
                        ] >= Ip + t
                    ) >= self.N or np.count_nonzero(
                            img[
                                y + self.circle_idx[0, :],
                                x + self.circle_idx[1, :]
                            ] <= Ip - t
                        ) >= self.N:
                        # Keypoint [corner]
                        keypoints.append([x, y])     # Note: keypoint = [col, row]
                        corner_img[y, x] = np.sum(
                            np.abs(
                                Ip - img[
                                    y + 
                                    self.circle_idx[0, :], x + 
                                    self.circle_idx[1, :]
                                ]
                              )
                        )

        return keypoints, corner_img
    
    def _apply_nms(self, raw_keypoints, corner_img):
        if self.nms_window == 0:
            return keypoints
        
        fewer_kps = []
        for [x, y] in raw_keypoints:
            window = corner_img[
                y - self.nms_window : y + self.nms_window + 1,
                x - self.nms_window : x + self.nms_window + 1
            ]

            loc_y_x = np.unravel_index(window.argmax(), window.shape)

            x_new = x + loc_y_x[1] - self.nms_window
            y_new = y + loc_y_x[0] - self.nms_window

            new_kp = [x_new, y_new]

            if new_kp not in fewer_kps:
                fewer_kps.append(new_kp)

        return fewer_kps


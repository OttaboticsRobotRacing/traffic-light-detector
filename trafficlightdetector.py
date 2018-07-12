import cv2
import numpy as np

class TrafficLightDetector:
    def __init__(self, reference_images, color_bounds, color_threshold=50, feature_threshold=10):
        self.reference_images = reference_images
        self.feature_threshold = feature_threshold
        self.color_threshold = color_threshold
        self.color_bounds = color_bounds

    def get_state(self, query):
        for reference in self.reference_images:
            traffic_light = self.feature_matching(query, reference)
            if traffic_light is not None:
                break

        if traffic_light is None:
            return None

        state = self.get_color(traffic_light)
        return state

    def get_color(self, traffic_light):
        hsv = cv2.cvtColor(traffic_light, cv2.COLOR_BGR2HSV)

        for color in self.color_bounds:
            bound = self.color_bounds[color]
            lower = np.array(bound[0])
            upper = np.array(bound[1])

            mask = cv2.inRange(hsv, lower, upper)

            # num_white = cv2.countNonZero(mask)
            num_white = np.sum(mask == 255)

            height, width = mask.shape
            num_pix = height*width

            percent = (num_white/num_pix)*100

            if percent > self.color_threshold:
                return color

        return None

    def feature_matching(self, query, reference):
        def get_ROI(img, points):
            """
            Returns image after being cropped to coordinates defined by the list points
            """
            x_list = []
            y_list = []
            for i in range(0,len(points),2):
                x_list.append(points[i])
            for i in range(1,len(points),2):
                y_list.append(points[i])

            tl = (min(x_list),min(y_list))
            br = (max(x_list),max(y_list))

            roi = img[tl[1]:br[1], tl[0]:br[0]]
            return roi

        def match_features(reference, query, min_match_count = 10):
            """
            Returns list of destination points
            """
            try:
                ref_grey = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            except:
                ref_grey = reference
            try:
                query_grey = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
            except:
                query_grey = query

            sift = cv2.xfeatures2d.SIFT_create()

            kp1, des1 = sift.detectAndCompute(query_grey,None)
            kp2, des2 = sift.detectAndCompute(ref_grey,None)

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1,des2,k=2)

            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)

            if len(good)>min_match_count:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = query_grey.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)

                dst_list = dst.ravel().tolist()
            else:
                dst_list = []

            for i,pnt in enumerate(dst_list):
                dst_list[i] = int(pnt)

            return dst_list

        dst_points = match_features(query, reference, self.feature_threshold)

        if dst_points is not None:
            return get_ROI(query, dst_points)

        return None

def main():
    pass

if __name__ == "__main__":
    main()

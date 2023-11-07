import numpy as np

# code from https://stackoverflow.com/a/61544442

def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    transMatrix = np.array([[0.299, 0.587, 0.114],
                            [0.59590059, -0.27455667, -0.32134392],
                            [0.21153661, -0.52273617, 0.31119955]]).transpose()
    shape = imgRGB.shape
    return np.dot(imgRGB.reshape(-1, 3), transMatrix).reshape(shape)

    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    transMatrix = np.linalg.inv(np.array([[0.299, 0.587, 0.114],
                                          [0.59590059, -0.27455667, -0.32134392],
                                          [0.21153661, -0.52273617, 0.31119955]])).transpose()
    shape = imgYIQ.shape
    return np.dot(imgYIQ.reshape(-1, 3), transMatrix).reshape(shape)
    pass


def hist_eq(img: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This function will do histogram equalization on a given 1D np.array
    meaning will balance the colors in the image.
    For more details:
    https://en.wikipedia.org/wiki/Histogram_equalization
    **Original function was taken from open.cv**
    :param img: a 1D np.array that represent the image
    :return: imgnew -> image after equalization, hist-> original histogram, histnew -> new histogram
    """

    # Flattning the image and converting it into a histogram
    histOrig, bins = np.histogram(img.flatten(), 256, [0, 255])
    # Calculating the cumsum of the histogram
    cdf = histOrig.cumsum()
    # Places where cdf = 0 is ignored and the rest is stored
    # in cdf_m
    cdf_m = np.ma.masked_equal(cdf, 0)
    # Normalizing the cdf
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # Filling it back with zeros
    cdf = np.ma.filled(cdf_m, 0)


    # Creating the new image based on the new cdf
    imgEq = cdf[img.astype('uint8')]
    histEq, bins2 = np.histogram(imgEq.flatten(), 256, [0, 256])

    return imgEq, histOrig, histEq

    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        The function will fist check if the image is RGB or gray scale
        If the image is gray scale will equalizes
        If RGB will first convert to YIQ then equalizes the Y level
        :param imgOrig: Original Histogram
        :return: imgnew -> image after equalization, hist-> original histogram, histnew -> new histogram
    """

    if len(imgOrig.shape) == 2:

        img = imgOrig * 255
        imgEq, histOrig, histEq = hist_eq(img)

    else:
        img = transformRGB2YIQ(imgOrig)
        img[:, :, 0] = img[:, :, 0] * 255
        img[:, :, 0], histOrig, histEq = hist_eq(img)
        img[:, :, 0] = img[:, :, 0] / 255
        imgEq = transformYIQ2RGB(img)

    return imgEq, histOrig, histEq
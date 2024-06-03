import numpy as np

# code from https://stackoverflow.com/a/61544442
# easy demonstration https://en.wikipedia.org/wiki/Histogram_equalization#Full-sized_image
def hist_eq(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    '''
    print(f'histOrig.cumsum: {histOrig.cumsum()}')
    print(f'cdf_new: {histEq.cumsum()}')
    print(f'histOrig.cumsum shape: {histOrig.cumsum().shape}')
    print(f'cdf_new shape: {histEq.cumsum().shape}')
    print(f'cdf: {cdf}')
    print(f'cdf shape: {cdf.shape}')

    print(f'num pixels with value 0 in original image: {np.sum(img == 0)}')
    print(f'num pixels with value 0 in equalized image: {np.sum(imgEq == 0)}')

    print(f'imgEq cumsum: {imgEq.sum() / 255}')
    print(f'img cumsum: {img.sum() / 255}')
    assert imgEq.sum() == img.sum(), 'final cumsum (overall image intensity) is equal'
    '''

    return imgEq, histOrig, histEq
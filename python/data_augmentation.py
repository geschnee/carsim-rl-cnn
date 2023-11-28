
import numpy as np

def salt_and_pepper_noise(image, prob=0.05):
    '''
    Adds salt and pepper noise to an image
    prob: probability (threshold) that controls level of noise
    '''
    output = np.copy(image)
    
    r = np.random.rand(*image.shape)
    print(f'r shape {r.shape}')

    prob_white = prob / 2
    prob_black = prob / 2

    output[r < prob_white] = 255 # white noise
    output[r > 1 - prob_black] = 0 # black noise

    return output

def gaussian_noise(image, mean=0, sigma=0.05):
    '''
    Adds gaussian noise to an image
    mean: mean of the noise
    sigma: standard deviation of the noise
    '''
    output = np.copy(image)
    noise = np.random.normal(mean, sigma, image.shape)

    print(f'noise max {np.max(noise)}')
    print(f'noise min {np.min(noise)}')
    output = output + noise
    output = np.clip(output, 0, 255)
    output = output.astype(np.uint8)
    return output
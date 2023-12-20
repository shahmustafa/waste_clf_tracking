import os
import cv2

organic = '/mnt/vol_1/waste_dataset/Mixed_Class/Segregated/organic'
plastic = '/mnt/vol_1/waste_dataset/Mixed_Class/Segregated/plastic'

mix = '/mnt/vol_1/waste_dataset/Mixed_Class/Mixed'
n = 1
for file_dry, file_wet in zip(os.listdir(organic), os.listdir(plastic)):

    img_dry = cv2.imread(os.path.join(organic, file_dry))
    img_wet = cv2.imread(os.path.join(plastic, file_wet))
    img_dry, img_wet = cv2.resize(img_dry, (500, 500)), cv2.resize(img_wet, (500, 500))

    # Define the region to splice from each image
    x1, y1, w1, h1 = 0, 0, img_dry.shape[1] // 2, img_dry.shape[0]
    x2, y2, w2, h2 = img_wet.shape[1] // 2, 0, img_wet.shape[1] // 2, img_wet.shape[0]

    # Extract the regions from the source images
    splice1 = img_dry[y1:y1 + h1, x1:x1 + w1]
    splice2 = img_wet[y2:y2 + h2, x2:x2 + w2]

    # Create a blank canvas for the new image
    img_mix = img_dry.copy()

    #     # Place the spliced regions onto the blank canvas
    #     img_mix[y1:y1+h1, x1:x1+w1] = splice2
    #     img_mix[y2:y2+h2, x2:x2+w2] = splice1

    # To inject some randomness in creating mixed images
    # Place the spliced regions onto the blank canvas
    if bool(n % 2):
        img_mix[y1:y1 + h1, x1:x1 + w1] = splice2
        img_mix[y2:y2 + h2, x2:x2 + w2] = splice1
    else:
        img_mix[y1:y1 + h1, x1:x1 + w1] = splice1
        img_mix[y2:y2 + h2, x2:x2 + w2] = splice2

    # Save the mix image
    cv2.imwrite(os.path.join(mix, ('mixed_' + str(n) + '.jpg')), img_mix)
    n += 1

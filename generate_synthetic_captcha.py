from utils.metrics import all_symbols, captcha_len, img_h, img_w
import os
from tqdm import tqdm
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2 as cv


def generate_captcha_images(train_num_images=100000, valid_num_images=10000, synthetic_folder='synthetic_datasets'):
    back = [[192 + i]*(3 + int(i <= 19)) for i in range(0, 60)]
    back = np.array([item for sublist in back for item in sublist])
    background = np.repeat(np.tile(back, (img_h, 1))[:, :, None], 3, axis=2).astype(np.uint8)
    font = ImageFont.truetype("Arial Bold Italic.ttf", 38)

    for num_images, dataset_folder in zip([train_num_images, valid_num_images],
                                          ['train_synthetic', 'valid_synthetic']):
        synthetic_dataset_path = os.path.join(synthetic_folder, dataset_folder)
        os.makedirs(synthetic_dataset_path, exist_ok=True)
        for _ in tqdm(range(num_images)):
            captcha_symbols = np.random.choice(all_symbols, captcha_len)
            image_pil = Image.fromarray(background)
            draw = ImageDraw.Draw(image_pil)
            x0 = 30
            y0 = 5
            for s in captcha_symbols:
                draw.text((x0, y0), s, font=font, fill=0)
                w0, h0 = font.getsize(s)
                x0 = x0 + w0 + np.random.randint(-8, 2)
                y0 = 5 + np.random.randint(-2, 4)

            image_pil = cv.blur(np.array(image_pil), (np.random.randint(1, 5), np.random.randint(1, 5)))
            xr0 = np.random.randint(15, 20)
            yr0 = np.random.randint(25, 55)
            xr1 = np.random.randint(30, 35)
            yr1 = np.random.randint(120, 150)
            gauss_noise = np.random.normal(0, 50, (xr1 - xr0, yr1 - yr0)).astype(np.int64)
            for c in range(3):
                image_pil[xr0:xr1, yr0:yr1, c] = np.clip(image_pil[xr0:xr1, yr0:yr1, c] + gauss_noise, 0, 255)

            image_pil = Image.fromarray(image_pil)
            draw = ImageDraw.Draw(image_pil)
            draw.line([(np.random.randint(5, 30), np.random.randint(0, 30)),
                       (np.random.randint(120, 170), np.random.randint(20, img_h-1))],
                      width=np.random.randint(1, 3), fill=(0, 0, 0))

            draw.line([(np.random.randint(5, 30), np.random.randint(0, 25)),
                       (np.random.randint(40, 70), np.random.randint(25, img_h-1))],
                      width=np.random.randint(1, 4), fill=(0, 0, 0))

            image_pil.save(os.path.join(synthetic_dataset_path, ''.join(captcha_symbols)) + '.png')


if __name__ == '__main__':
    generate_captcha_images()


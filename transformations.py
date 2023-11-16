import torchvision.transforms as transforms
import random

def config_gettransformation(c, d):
    match c:
        case "CentreCrop":
            return CentreCrop(output_size= d)
        case "LeftCrop":
            return LeftCrop(output_size= d)
        case "RightCrop":
            return RightCrop(output_size= d)
        case "SampleNthPixel":
            return SampleNthPixel(n=d)
        case "RandomCrop":
            return RandomCrop(output_size = d)
        case _:
            raise Exception("Invalid Transformation")


class Transformation():
    def __init__(self, output_size):
        self.output_size = output_size
        self.default_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, input_img, ground_truth_img):
        return self.default_transforms(input_img), ground_truth_img

class CentreCrop(Transformation):
    def __call__(self, input_img, ground_truth_img):
        input_img, ground_truth_img = super().__call__(input_img, ground_truth_img)

        center_crop = transforms.CenterCrop(self.output_size)
        return center_crop(input_img), center_crop(ground_truth_img)
                               

class RandomCrop(Transformation):
    def __call__(self, input_img, ground_truth_img):
        input_img, ground_truth_img = super().__call__(input_img, ground_truth_img)

        width, height = input_img.shape[1], input_img.shape[2]
        new_height, new_width = self.output_size

        top = random.randint(0, height - new_height)
        left = random.randint(0, width - new_width)

        input_crop = input_img[:, top:top + new_height, left:left + new_width]
        ground_truth_crop = ground_truth_img[:, top:top + new_height, left:left + new_width]

        return input_crop, ground_truth_crop

class SampleNthPixel(Transformation):
    def __init__(self, n):
        super().__init__((None, None))
        self.n = n

    def __call__(self, input_img, ground_truth_img):
        input_img, ground_truth_img = super().__call__(input_img, ground_truth_img)

        input_sample = input_img[:, ::self.n, ::self.n]
        ground_sample = ground_truth_img[:, ::self.n, ::self.n]

        return input_sample, ground_sample

class LeftCrop(Transformation):
    def __call__(self, input_img, ground_truth_img):
        input_img, ground_truth_img = super().__call__(input_img, ground_truth_img)

        left_crop = lambda img: img.crop((0, 0, self.output_size[0], self.output_size[1]))
        return left_crop(input_img), left_crop(ground_truth_img)


class RightCrop(Transformation):
    def __call__(self, input_img, ground_truth_img):
        input_img, ground_truth_img = super().__call__(input_img, ground_truth_img)

        right_crop = lambda img: img.crop((img.width - self.output_size[0], 0, img.width, self.output_size[1]))
        return right_crop(input_img), right_crop(ground_truth_img)
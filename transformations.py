import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

def config_gettransformation(c, d):
    match c:
        case "CentreCrop":
            return CentreCrop(output_size= d)
        case "LeftCrop":
            return LeftCrop(output_size= d)
        case "RightCrop":
            return RightCrop(output_size= d)
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

        r = transforms.RandomCrop(self.output_size)
        i, j, h, w = r.get_params(input_img, r.size)

        return transforms.functional.crop(input_img, i, j, h, w), transforms.functional.crop(ground_truth_img, i, j, h, w)

class LeftCrop(Transformation):
    def __call__(self, input_img, ground_truth_img):
        input_img, ground_truth_img = super().__call__(input_img, ground_truth_img)

        _, width, height = input_img.shape
        crop_width = self.output_size
        crop_height = self.output_size
        left = 0
        bottom = height - crop_height

        crop_width = min(crop_width, width)
        crop_height = min(crop_height, height)

        left_cropped_input = TF.crop(input_img, top=bottom, left=left, height=crop_height, width=crop_width)
        left_cropped_ground_truth = TF.crop(ground_truth_img, top=bottom, left=left, height=crop_height, width=crop_width)

        return left_cropped_input, left_cropped_ground_truth



class RightCrop(Transformation):
    def __call__(self, input_img, ground_truth_img):
        input_img, ground_truth_img = super().__call__(input_img, ground_truth_img)

        _, width, height = input_img.shape
        crop_width = self.output_size
        crop_height = self.output_size
        right = width  
        bottom = height - crop_height

        crop_width = min(crop_width, width)
        crop_height = min(crop_height, height)

        left = max(0, right - crop_width)

        right_cropped_input = TF.crop(input_img, top=bottom, left=left, height=crop_height, width=crop_width)
        right_cropped_ground_truth = TF.crop(ground_truth_img, top=bottom, left=left, height=crop_height, width=crop_width)

        return right_cropped_input, right_cropped_ground_truth
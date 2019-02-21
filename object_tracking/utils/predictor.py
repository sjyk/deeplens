from torchvision import transforms
from torch.autograd import Variable
import torch
from PIL import Image

from .general_utils import non_max_suppression

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor


class Predictor:

    def __init__(self, args, model):
        self.args = args
        self.model = model

    def detect_image(self, img):
        # scale and pad image
        ratio = min(self.args.img_size / img.size[0],
                    self.args.img_size / img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                             transforms.Pad((max(
                                                 int((imh - imw) / 2), 0), max(
                                                 int((imw - imh) / 2), 0), max(
                                                 int((imh - imw) / 2), 0), max(
                                                 int((imw - imh) / 2), 0)),
                                                 (128, 128, 128)),
                                             transforms.ToTensor(),
                                             ])
        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(Tensor))
        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections,
                                             self.args.num_classes,
                                             self.args.conf_thres,
                                             self.args.nms_thres)
        return detections[0]

    def predict_for_mot(self, frame):
        pilimg = Image.fromarray(frame)
        return self.detect_image(pilimg).cpu()

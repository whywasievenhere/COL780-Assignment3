import os
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import detection
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default='./', help="path to dataset root directory")
parser.add_argument("--test", type=str, default='PennFudanPed_val.json', help="path to test json")
parser.add_argument("--out", type=str, default='frcnn_output_file.json', help="path to output json")
parser.add_argument("--model", default=None, help="path to pretrained Faster RCNN weights file")

args = parser.parse_args()

with open(args.test) as f:
    data = json.load(f)

imgs = [data['images'][i]['file_name'] for i in range(len(data['images']))]
masks = [[] for i in range(len(data['images']))]
for i in range(len(data['annotations'])):
    tmp = data['annotations'][i]['bbox']
    tmp = [int(x) for x in tmp]
    tmp = [tmp[0], tmp[1], tmp[0]+tmp[2], tmp[1]+tmp[3]]
    masks[data['annotations'][i]['image_id']].append(tmp)

class PedestrainDataset(Dataset):
    def __init__(self, root, transforms, imgs, masks):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        self.imgs = imgs
        self.masks = masks

    def __getitem__(self, idx):
        # # load images and masks
        # img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        # img = Image.open(img_path).convert("RGB")
        # orig = cv2.imread(img_path)
        # # note that we haven't converted the mask to RGB,
        # # because each color corresponds to a different instance
        # # with 0 being background
        # mask = Image.open(mask_path)
        # # convert the PIL Image into a numpy array
        # mask = np.array(mask)
        # # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        orig = cv2.imread(img_path)

        # # split the color-encoded mask into a set
        # # of binary masks
        # masks = mask == obj_ids[:, None, None]

        # # get bounding box coordinates for each mask
        # num_objs = len(obj_ids)
        # boxes = []
        # for i in range(num_objs):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])

        boxes = self.masks[idx]
        num_objs = len(self.masks[idx])
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["original"] = orig
        out_path = self.imgs[idx].split('/')
        out_path[1] = 'output'
        out_path = '/'.join(out_path)
        target["output_path"] = os.path.join(self.root, out_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize a dictionary containing model name and its corresponding 
# torchvision function call
MODELS = {
	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	"retinanet": detection.retinanet_resnet50_fpn
}
# load the model and set it to evaluation mode
MODEL_NAME = "frcnn-resnet"
model = MODELS[MODEL_NAME](pretrained=True, progress=True, pretrained_backbone=True).to(device) if args.model is None else torch.load(args.model)
# torch.save(model, './weights/{}.pth'.format(MODEL_NAME))
model.eval()

# Convert Image to tensor (can be used for other transforms compositionally as well).
import torchvision.transforms as T
transforms = T.Compose(
    [
        T.ToTensor(),
    ]
)

COLOR = 5
SAVE_PREDICTIONS = True
def main(threshold=0.9):
    eval_dataset = PedestrainDataset(args.root, transforms, imgs, masks)

    data_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    json_dump = []

    img_number = -1
    with torch.no_grad():
        for input, target in data_loader:
            img_number += 1
            input = input.to(device)

            outputs = model(input)[0]

            img = np.squeeze(target["original"].cpu().detach().numpy())
            for i in range(len(outputs["boxes"])):
                confidence = outputs["scores"][i].item()

                # If not pedestrian, ignore object!
                idx = int(outputs["labels"][i])
                if idx != 1: 
                    continue

                box = outputs["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int").tolist()

                current_json = {}
                current_json["image_id"] = img_number
                current_json["category_id"] = 1
                current_json["bbox"] = (startX, startY, endX-startX, endY-startY)
                current_json["score"] = confidence
                json_dump.append(current_json)

                if confidence > threshold:
                    label = "{}: {:.2f}%".format("Pedestrian", confidence * 100)
                    cv2.rectangle(img, (startX, startY), (endX, endY), COLOR, 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(img, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
            
            if SAVE_PREDICTIONS:
                cv2.imwrite(target["output_path"][0], img)

    with open(args.out, 'w') as fout:
        json.dump(json_dump, fout)

if __name__ == '__main__':
    main()
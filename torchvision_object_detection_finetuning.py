import matplotlib.pyplot as plt
from torchvision.io import read_image
import os
import torch
import torchvision

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

os.system(
    "wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py"
)
os.system(
    "wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py"
)
os.system(
    "wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py"
)
os.system(
    "wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py"
)
os.system(
    "wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py"
)

import utils
from engine import train_one_epoch, evaluate

# cargamos una imagen de muestra del dataset y su máscara
image = read_image("PennFudanPed/PNGImages/FudanPed00046.png")
mask = read_image("PennFudanPed/PedMasks/FudanPed00046_mask.png")

# mostramos el ejemplo con ámbas imágenes
plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title("Image")
plt.imshow(image.permute(1, 2, 0))
plt.subplot(122)
plt.title("Mask")
plt.imshow(mask.permute(1, 2, 0))
plt.show()


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # listamos y ordenamos las imágenes y las máscaras
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # cargamos imágen con la máscara correspondiente
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = read_image(img_path)
        mask = read_image(mask_path)

        # extraemos los id
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # convertimos el id en máscaras binarias
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # generamos las cajas a partir de la máscara
        boxes = masks_to_boxes(masks)

        # etiquetas de la clase persona
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # convertimos la imagen a tensor de torchvision
        img = tv_tensors.Image(img)

        # creamos el diccionario target con todas las anotaciones para entrenar el modelo
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F.get_size(img)
        )
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # aplicamos transformaciones
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# cargamos el modelo preentrenado en COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

num_classes = 2

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# modificamos el modelo para añadir un backbone diferente

backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features

backbone.out_channels = 1280

anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0))
)

roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=["0"], output_size=7, sampling_ratio=2
)

model = FasterRCNN(
    backbone,
    num_classes=2,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
)


def get_model_instance_segmentation(num_classes):

    # creamos el modelo mask r-cnn
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # obtenemos el número de características que entran a la capa final
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # reemplazamos el box predictor con uno nuevo
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # obtenemos el número de canales de entreada del predictor de máscaras
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # lo reemplazamos por uno adaptado al número de clases
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


# agregamos transformaciones a las imágenes como invertir horizontalmente la imagen, máscara y boxes
# con una probabilidad del 50%
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))  # convertimos a tipo float
    transforms.append(T.ToPureTensor())  # convertimos a tensor puro
    return T.Compose(transforms)


# creamos device para usar la  gpu con cuda sino usamos la cpu
device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)

# número de clases (persona y fondo)
num_classes = 2

# cargamos las imágenes y máscaras del dataset
dataset = PennFudanDataset("PennFudanPed", get_transform(train=True))
dataset_test = PennFudanDataset("PennFudanPed", get_transform(train=False))

# dividimos en el set de entrenamiento y el de test
# todas menos las 50 últimas de cada set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# creamos los dataloader de entrenamiento
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn
)

# creamos el dataloader de test
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn
)

# cargamos el modelo mask r-cnn ajustado
model = get_model_instance_segmentation(num_classes)

# enviamos el modelo a la gpu
model.to(device)

# parámetros a entrenar
params = [p for p in model.parameters() if p.requires_grad]

# definimos el optimizador
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# programamos la tasa de aprendizaje
# cada 3 epochs el lr se multiplica por 0.1
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# entrenamiento
num_epochs = 2

for epoch in range(num_epochs):

    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

    lr_scheduler.step()

    evaluate(model, data_loader_test, device=device)


image = read_image("PennFudanPed/PNGImages/FudanPed00046.png")
eval_transform = get_transform(train=False)

# evaluación
model.eval()
with torch.no_grad():
    x = eval_transform(image)

    x = x[:3, ...].to(device)
    predictions = model(
        [
            x,
        ]
    )
    pred = predictions[0]

# normalizamos la imagen
image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]
pred_labels = [
    f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])
]
# dibujamos las cajas
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

# dibujamos las máscaras
masks = (pred["masks"] > 0.7).squeeze(1)
output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))
plt.show()

import torch

def iou_single_class(pred, target):
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()

    if union == 0:
        return 1.0

    iou = intersection / union
    return iou

def mean_iou(output, target, num_classes):
    preds = torch.argmax(output, dim=1)
    total_iou = 0.0
    for cls in range(num_classes):
        class_pred = (preds == cls)
        class_target = (target == cls)
        total_iou += iou_single_class(class_pred, class_target)
    return total_iou / num_classes

def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    total_mIoU = 0.0

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            print(f"{i}/{len(dataloader)}")
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_mIoU += mean_iou(outputs, masks, num_classes)

    mean_mIoU = total_mIoU / len(dataloader)
    return mean_mIoU
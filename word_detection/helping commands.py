python -m torch.distributed.launch --nproc_per_node 4 train.py --weights yolov5s.pt --data my_dataset/my_dataset.yaml --epochs 300
--batch-size 256 --workers 32 --cache disk --patience 100 --device 0,1,
python -m torch.distributed.launch --nproc_per_node 4 train_gnhk_4_channel.py --weights yolov5m6.pt --cfg ./models/yolov5m.yaml --data ./data/gnhk_all.yaml --epochs 150 --batch-size 8 --workers 4 --cache None --patience 100 --device 1,2,5,6 --imgsz 2560


def pull_item(self, index):
    # if self.get_imagename(index) == 'img_59.jpg':
    #     pass
    # else:
    #     return [], [], [], [], np.array([0])
    # image, character_bboxes, words, confidence_mask, confidences = self.load_image_gt_and_confidencemask(index)
    image, character_bboxes, words, confidence_mask, confidences, target = self.load_image_gt_and_confidencemask(index)
    # , target
    if len(confidences) == 0:
        confidences = 1.0
    else:
        confidences = np.array(confidences).mean()
    region_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    affinity_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    affinity_bboxes = []

    if len(character_bboxes) > 0:
        region_scores = self.gaussianTransformer.generate_region(region_scores.shape, character_bboxes)
        affinity_scores, affinity_bboxes = self.gaussianTransformer.generate_affinity(region_scores.shape,
                                                                                      character_bboxes,
                                                                                      words)
    if self.viz:
        self.saveImage(self.get_imagename(index), image.copy(), character_bboxes, affinity_bboxes, region_scores,
                       affinity_scores,
                       confidence_mask)
    """    
    random_transforms = [image, region_scores, affinity_scores, confidence_mask]
    random_transforms = random_crop(random_transforms, (self.target_size, self.target_size), character_bboxes, target)
    #random_transforms = random_horizontal_flip(random_transforms)
    #random_transforms = random_rotate(random_transforms)
    cvimage, region_scores, affinity_scores, confidence_mask = random_transforms
    """
    bboxes_list = target  # bboxes.read().split('\n')
    num_objs = len(bboxes_list)
    boxes = []
    class_ = 0
    for i in range(num_objs):
        # b = bboxes_list[i].split(',')
        b = bboxes_list[i]
        x1 = max(min(int(b[0][0]), int(b[1][0]), int(b[2][0]), int(b[3][0])), 0)
        y1 = max(min(int(b[0][1]), int(b[1][1]), int(b[2][1]), int(b[3][1])), 0)
        ### FIXME: check the image shape for x and y coordinates???? :)
        x2 = min(max(int(b[0][0]), int(b[1][0]), int(b[2][0]), int(b[3][0])), image.shape[1])
        y2 = min(max(int(b[0][1]), int(b[1][1]), int(b[2][1]), int(b[3][1])), image.shape[0])
        boxes.append([x1, y1, x2, y2])
    category_ids = [1] * len(boxes)  # target.shape[0]
    transformalbum = A.Compose([
        A.RandomCrop(height=self.target_size, width=self.target_size, p=1, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
    ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
        additional_targets={'region_scores': 'image', 'affinity_scores': 'image', 'confidence_mask': 'image',
                            'bboxes': 'boxes'}
    )

    transformed = transformalbum(image=image, region_scores=region_scores, affinity_scores=affinity_scores,
                                 confidence_mask=confidence_mask,
                                 bboxes=boxes, category_ids=category_ids)

    cvimage = transformed['image']
    region_scores = transformed['region_scores']
    affinity_scores = transformed['affinity_scores']
    confidence_mask = transformed['confidence_mask']
    target = transformed['bboxes']
    ### FIXME: x_center, y_center, width, height
    target_normalized = []
    img_width, img_height = image.shape[0], image.shape[1]
    for i in range(num_objs):
        x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
        width, height = x2 - x1, y2 - y1
        x_center, width = round(x_center / img_height, 6), round(width / img_height, 6)
        y_center, height = round(y_center / img_width, 6), round(height / img_width, 6)
        target_normalized.append([class_, x_center, y_center, width, height])

    region_scores = self.resizeGt(region_scores)
    affinity_scores = self.resizeGt(affinity_scores)
    confidence_mask = self.resizeGt(confidence_mask)

    if self.viz:
        self.saveInput(self.get_imagename(index), cvimage, region_scores, affinity_scores, confidence_mask)
    image = Image.fromarray(cvimage)
    image = image.convert('RGB')
    image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)

    image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
                                          variance=(0.229, 0.224, 0.225))
    image = torch.from_numpy(image).float().permute(2, 0, 1)
    region_scores_torch = torch.from_numpy(region_scores / 255).float()
    affinity_scores_torch = torch.from_numpy(affinity_scores / 255).float()
    confidence_mask_torch = torch.from_numpy(confidence_mask).float()
    # return image, region_scores_torch, affinity_scores_torch, confidence_mask_torch, confidences
    """
    bboxes_list = target  # bboxes.read().split('\n')
    num_objs = len(bboxes_list)
    boxes = []
    for i in range(num_objs):

        # b = bboxes_list[i].split(',')
        b = bboxes_list[i]

        boxes.append([b[0], b[1], b[2], b[3]])
    """
    target_normalized = target_normalized
    target_normalized = np.asarray(target_normalized) * 1 / 2
    num_objs = len(boxes)
    target_normalized = torch.as_tensor(target_normalized, dtype=torch.float32)
    # there is only one class
    """
    labels = torch.ones((num_objs,), dtype=torch.int64)

    image_id = torch.tensor([index])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    # suppose all instances are not crowd
    iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    # target["masks"] = masks
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd
    """
    return image, region_scores_torch, affinity_scores_torch, confidence_mask_torch, confidences, target_normalized

_, out, train_out, _ = model(im, False) if training else model(im,False, augment=augment, val=True)
if epoch % 50 == 0 and epoch != 0:
    step_index += 1
    ###FIXME: check the value of lr adjustment for craft branch
    adjust_learning_rate(optimizer, hyp, step_index)
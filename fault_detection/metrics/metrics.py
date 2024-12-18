class AverageMeter:
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def compute_iou(im1, im2):
    tp = ((im1 > 0.5) & (im2 > 0.5)).sum()  
    tn = ((im1 <= 0.5) & (im2 <= 0.5)).sum()
    return tp / (im1.numel() - tn)


def compute_dice(im1, im2):
    tp = ((im1 > 0.5) & (im2 > 0.5)).sum()  
    tn = ((im1 <= 0.5) & (im2 <= 0.5)).sum()
    return 2 * tp / (im1.numel() - tn + tp)


def compute_acc(im1, im2):
    tp = ((im1 > 0.5) & (im2 > 0.5)).sum()  
    tn = ((im1 <= 0.5) & (im2 <= 0.5)).sum()
    return (tp + tn) / im1.numel()


def compute_prec(im1, im2):
    tp = ((im1 > 0.5) & (im2 > 0.5)).sum()  
    fp = ((im1 <= 0.5) & (im2 > 0.5)).sum()  
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def compute_recall(im1, im2):
    tp = ((im1 > 0.5) & (im2 > 0.5)).sum()  
    fn = ((im1 > 0.5) & (im2 <= 0.5)).sum()  
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def compute_f1(im1, im2):
    prec = compute_prec(im1, im2)
    recall = compute_recall(im1, im2)
    return 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0



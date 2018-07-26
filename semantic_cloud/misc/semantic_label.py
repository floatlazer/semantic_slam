import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import numpy as np
'''
Draw semantic class label colors
'''

def color_map(N=256, normalized=False):
    """
    Return Color Map in PASCAL VOC format
    """
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = (r, g, b)
    cmap = cmap/255.0 if normalized else cmap
    return cmap

# SUNRGBD
n_classes_sunrgbd = 38
labels_sunrgbd = ['backgroud', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
        'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror',
        'floor_mat', 'clothes', 'ceiling', 'books', 'fridge', 'tv', 'paper', 'towel', 'shower_curtain',
        'box', 'whiteboard', 'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag']

# Pascal VOC
n_classes_pascal = 21
labels_pascal = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog',
        'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

# ADE20K
# Reference: https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.txt
n_classes_ade20k = 150
labels_ade20k = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']

def draw_labels(n_classes, labels, labels_per_col, rect_width = 12, rect_height = 4, fontsize = 9):
    '''
    draw labels
    '''
    # Plot
    fig, ax = plt.subplots(nrows  = 1, ncols = 1, tight_layout = True)
    colors = color_map(N = n_classes, normalized = True)
    rectangles = {}
    # Make rects
    for i in range(n_classes):
        col = i // labels_per_col
        row = i % labels_per_col
        rectangles[labels[i]] =  mpatch.Rectangle((col * rect_width, row * rect_height), rect_width, rect_height, facecolor = colors[i])

    for r in rectangles:
        ax.add_artist(rectangles[r])
        rx, ry = rectangles[r].get_xy()
        cx = rx + rectangles[r].get_width()/2.0
        cy = ry + rectangles[r].get_height()/2.0

        ax.annotate(r, (cx, cy), color='w', weight='semibold',
                    fontsize=fontsize, ha='center', va='center')
    ax.axis('off')
    ax.set_xlim((0, rect_width * n_classes // labels_per_col))
    ax.set_ylim((0, rect_height * labels_per_col))
    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    #draw_labels(n_classes_pascal, labels_pascal, labels_per_col = 21)
    #draw_labels(n_classes_sunrgbd, labels_sunrgbd, labels_per_col = 19)
    draw_labels(n_classes_ade20k, labels_ade20k, labels_per_col = 15, rect_height = 4, rect_width = 10)

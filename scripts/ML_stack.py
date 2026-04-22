import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import cv2
import ultralytics
from segment_anything import sam_model_registry

print("torch:", torch.__version__)
print("cv2:", cv2.__version__)
print("ultralytics:", ultralytics.__version__)
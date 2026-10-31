"""Manual GUI preview (not a unit test): original | enhanced, side by side."""
import sys

import cv2

from extreme_cond_enhancer import LightingEnhancer

IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "extreme_cond_enhancer/tests/sample.jpeg"

enhancer = LightingEnhancer()

img = cv2.imread(IMG_PATH)  # BGR, as the enhancer expects
if img is None:
    sys.exit(f"cannot read {IMG_PATH}")
output, meta = enhancer.enhance(img)
print(meta)
preview = cv2.hconcat([img, output])
scale = min(1.0, 1400 / preview.shape[1])
cv2.imshow("original | enhanced", cv2.resize(preview, None, fx=scale, fy=scale))
cv2.waitKey(0)
cv2.destroyAllWindows()

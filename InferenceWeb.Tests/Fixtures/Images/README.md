`edit-photo-1024x768.heic` is a generated test image with a red rectangle and
colored corner markers, encoded as HEVC in HEIF with Pillow and pillow-heif.
Its 1024 by 768 canvas exceeds the chat thumbnail area so the upload tests can
verify that precise editing receives the original dimensions and decoded pixels.
Tests only require HEIC decoding; they do not require an HEIC encoder.

To regenerate with Pillow and pillow-heif:

```python
from PIL import Image, ImageDraw
import pillow_heif

pillow_heif.register_heif_opener()
image = Image.new('RGB', (1024, 768), (220, 220, 220))
draw = ImageDraw.Draw(image)
draw.rectangle((384, 280, 640, 520), fill=(190, 30, 30))
draw.rectangle((0, 0, 63, 63), fill=(30, 180, 60))
draw.rectangle((960, 704, 1023, 767), fill=(40, 70, 200))
image.save('edit-photo-1024x768.heic', format='HEIF', quality=90)
```

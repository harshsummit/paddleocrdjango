from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import io
import json
import base64
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def index(request):
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False) 
    request_body = request.body.decode('utf-8')
    json_data = json.loads(request_body)
    img = converB64tofile(json_data['file'])
    result = ocr.ocr(img, cls=True)
    return HttpResponse(result)

def converB64tofile(b64):
  image = np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
  return image
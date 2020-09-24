from django.http import HttpResponse
from django.http import StreamingHttpResponse
from cv2 import cv2

def test1(response):
    return HttpResponse("Hy I am  here")

def startStream(request):
    
    response = StreamingHttpResponse(request)
    print(response._convert_to_charset)
    return response
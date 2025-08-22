# myshoplifterproject/prediction/views.py
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .predictor import predict_video
import os

def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video_file'):
        video_file = request.FILES['video_file']
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        uploaded_file_path = fs.path(filename)
        
        prediction = predict_video(uploaded_file_path)
        
        fs.delete(filename)
        
        return render(request, 'prediction/result.html', {'prediction': prediction})
    return render(request, 'prediction/upload.html')
import time

from django.contrib.auth.models import User
from django.conf import settings
from django.core.files.uploadhandler import TemporaryFileUploadHandler
from django.shortcuts import get_object_or_404
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt
from django.http.multipartparser import MultiPartParser
from django.views.decorators.http import require_POST
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.cache import cache
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer
from django.http import JsonResponse, FileResponse
from django.http.multipartparser import MultiPartParser
from .modules.MAL_utils import get_data
from .modules.general_utils import determine_queue_cache_key
from .tasks import get_user_seasonal_stats_task, get_user_recs_task, get_user_affs_task
from .models import AnimeDataUpdated, SavedImage
import requests
import json
import logging
import os
from urllib.parse import unquote
from celery.result import AsyncResult
from django.db.utils import OperationalError

logger = logging.getLogger('nyanpasutats')

CACHE_TIMEOUT = 3600
TASK_TIMEOUT = 3600


# class TokenObtainView(APIView):
#     permission_classes = (AllowAny,)
#
#     def get(self, request):
#         # For simplicity, assuming the user is already authenticated
#         # Here, generate the token for an existing user (e.g., during login)
#         user = User.objects.get(username=request.data['username'])
#         refresh = RefreshToken.for_user(user)
#
#         return Response({
#             'refresh': str(refresh),
#             'access': str(refresh.access_token),
#         })



def get_task_data(request):
    logger.info("Entered get task data")
    task_id = request.GET.get('task_id')
    task_result = AsyncResult(task_id)
    time_elapsed = 0
    while time_elapsed < TASK_TIMEOUT:
        try:
            if task_result.ready():
                result = task_result.get()
                if type(result) == dict and 'error' in result.keys():
                    return JsonResponse({'status': 'error', 'error': result['error']}, status=result['status'])
                return JsonResponse({'status': 'completed', 'data': result}, status=200)
        except OperationalError:
            logger.error(f"Caught Operational Error in task {task_id}.")
        time.sleep(1)
        time_elapsed += 1
        if time_elapsed % 120 == 0:
            logger.warning(f"Warning - Task {task_id} is incomplete after {time_elapsed/60} minutes.")

    logger.error(f"Error - Task {task_id} was unable to complete on time.")
    return JsonResponse({'status': 'error',
                         'data': f'Task {task_id} was unable to complete on time.'}, status=500)


class BaseSectionView(APIView):
    task_function = None

    def get(self, request):
        username = request.query_params.get('username')
        site = request.query_params.get('site')

        if not username:
            return Response({"error": "Username is required"}, status=400)

        cache_key = f"{self.task_function.__name__}_{username}_{site}_"
        result = cache.get(cache_key)
        if result is not None:
            logger.info(f"Cached result for {username}'s task found")
            return Response({'status': 'completed', 'data': result}, status=200)

        task = self.task_function.delay(username, site)
        return Response({'taskId': task.id}, status=202)


class RecommendationsView(BaseSectionView):
    task_function = get_user_recs_task


class AffinityFinderView(BaseSectionView):
    task_function = get_user_affs_task


class SeasonalStatsView(BaseSectionView):
    task_function = get_user_seasonal_stats_task


@api_view(('GET',))
@renderer_classes((JSONRenderer,))
def get_anime_img_url(request):
    show_name = unquote(request.GET.get('show_name', ''))
    try:
        image_url = AnimeDataUpdated.objects.get(name=show_name).image_url
        return Response(image_url)
    except AnimeDataUpdated.DoesNotExist:
        return Response("Show name does not exist.", status=400)


@api_view(('GET',))
@renderer_classes((JSONRenderer,))
def get_queue_position(request):
    random = request.GET.get('random')
    if not random:
        return Response({'error': "Unauthorized access"}, status=403)

    queue_type = request.GET.get('type')
    queue_cache_key = determine_queue_cache_key(queue_type, "view")
    return Response({'queuePosition': cache.get(queue_cache_key, 1)}, status=200)


@api_view(('GET',))
@renderer_classes((JSONRenderer,))
def get_anime_img_urls(request):
    show_names = request.GET.get('show_names', '[]')
    logger.info(f"Requested images for {show_names}")
    try:
        show_names_list = json.loads(show_names)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON format'}, status=400)
    try:
        # Replace this with parquet after adding image_url to parquet
        # in next database iteration
        anime_data = [AnimeDataUpdated.objects.get(name=show) for show in show_names_list]
        image_urls = [data.image_url for data in anime_data]

        if len(image_urls) != len(show_names_list):
            # Find the missing one and replace it with a placeholder empty string
            anime_names_with_images = [data['name'] for data in anime_data]
            image_urls = [data['image_url'] if data['name'] in anime_names_with_images else ''
                          for data in anime_data]

        return Response(image_urls)
    except AnimeDataUpdated.DoesNotExist:
        logger.error("One or more of the shows requested does not exist.")
        return Response("One or more of the shows requested does not exist.", status=400)


@require_POST
def update_image_url_view(request):
    data = json.loads(request.body)
    show_name = data.get('show_name')
    logger.info(f"Requested images for {show_name}")

    try:
        existing_show_data = AnimeDataUpdated.objects.get(name=show_name)
    except AnimeDataUpdated.DoesNotExist:
        return JsonResponse({'added': False, 'error': 'Show does not exist'}, status=400)

    show_id = existing_show_data.mal_id
    existing_image_url = existing_show_data.image_url
    image_exists = requests.head(existing_image_url)
    if not image_exists:
        url = f'https://api.myanimelist.net/v2/anime/{show_id}?fields=main_picture'
        MAL_show_data = get_data(url)
        image_url = MAL_show_data['main_picture']['medium']
        existing_show_data.image_url = image_url
        existing_show_data.save()
        return JsonResponse({'added': True})
    return JsonResponse({'added': False})


@csrf_exempt
def upload_infographic_image(request):
    if request.method == 'POST':

        upload_handlers = [TemporaryFileUploadHandler()]
        parser = MultiPartParser(request.META, request, upload_handlers)
        parsed_data, files = parser.parse()
        if 'image' not in files:
            return JsonResponse({"error": "Image not sent"}, status=400)

        image_type = parsed_data.get('image_type')  # Adjust key as necessary
        if image_type not in ['tierList', 'seasonalCard']:
            return JsonResponse({"error": "Invalid image type"}, status=400)

        image_name = parsed_data.get('image_name')
        try:
            file_path = default_storage.save(f"userImages/{image_name}.png", files['image'])
            # This file_path might not be exactly f"userImages/{image_name}.png if the image
            # with that name already existed. In that case, a random string will be added to the name.
            # SomeImage.png ---> SomeImage-34SDMK189.png.
            saved_image = SavedImage(file_path=file_path, file_name=image_name)
            saved_image.save()
        except Exception as e:
            logger.error(f"An error has occurred while saving the image with name {image_name}. {e}")

        image_url = f"https://nps.moe/userImages/{saved_image.unique_id}"
        return JsonResponse({"url": image_url})
    else:
        return JsonResponse({"error": "No image provided"}, status=400)


def fetch_infographic_image(request):
    unique_id = request.GET.get('unique_id', "")
    saved_image = get_object_or_404(SavedImage, unique_id=unique_id)
    # Serve the file
    file_path = saved_image.file_path  # Full path to the file
    image_url = f"{settings.MEDIA_URL}{file_path}"
    try:
        return JsonResponse({"url": image_url})
    except Exception as e:
        logger.error(f"An error has occurred while fetching the image with url {image_url}. {e}")



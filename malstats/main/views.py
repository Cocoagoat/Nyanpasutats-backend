import time

from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .modules.Model import Model
from pathlib import Path
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.cache import cache
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer, TemplateHTMLRenderer
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator
from django.db.models import Case, When
from django.http import JsonResponse
from .modules.AffinityFinder import find_max_affinity
from .modules.Errors import UserListFetchError
from .modules.SeasonalStats import SeasonalStats
from .modules.MAL_utils import get_data
from .tasks import get_user_seasonal_stats_task, get_user_recs_task, get_user_affs_task
from .models import AnimeData, TaskQueue, UsernameCache, AnimeDataUpdated
from .modules.UserDB import UserDB
import logging
import requests
import json
from .modules.Log_config import *
import pdb
from urllib.parse import unquote
from celery.result import AsyncResult
from django.db.utils import OperationalError


view_logger = logging.getLogger('Nyanpasutats.view')
# class MyDataView(APIView):
#     def get(self, request, format=None):
#         data = your_python_script_function()  # This returns a list or dictionary
#         return Response(data)


# user_db = UserDB()
CACHE_TIMEOUT = 3600
TASK_TIMEOUT = 3600


def delete_task(task_id):
    task = TaskQueue.objects.get(task_id=task_id)
    task.delete()


def get_task_data2(request):
    task_id = request.GET.get('task_id')
    task_result = AsyncResult(task_id)

    if task_result.ready():
        result = task_result.get()
        # try:
        #     task = TaskQueue.objects.get(task_id=task_id)
        #     task.delete()
        # except TaskQueue.DoesNotExist:
        #     print("Task not found.")
        #     view_logger.error("Task not found")

        return JsonResponse({'status': 'completed', 'data': result}, status=200)
    else:
        return JsonResponse({'status': 'pending'}, status=202)


def get_task_data(request):
    print("Entered get task data")
    task_id = request.GET.get('task_id')
    task_result = AsyncResult(task_id)
    time_elapsed = 0
    # pdb.set_trace()
    while time_elapsed < TASK_TIMEOUT:
        try:
            if task_result.ready():
                try:
                    # pdb.set_trace()
                    result = task_result.get()
                    # delete_task(task_id)
                    if type(result) == dict and 'error' in result.keys():
                        return JsonResponse({'status': 'error', 'data': result['error']}, status=result['status'])
                    return JsonResponse({'status': 'completed', 'data': result}, status=200)
                except TaskQueue.DoesNotExist:
                    print("Task not found.")
                    # Find a better way to handle task not found in queue case
                    return JsonResponse({'status': 'error', 'data': 'Task not found.'}, status=400)
        except OperationalError:
            view_logger.error("Caught Operational Error")
        time.sleep(1)
        time_elapsed += 1
        if time_elapsed % 120 == 0:
            view_logger.warning(f"Warning - Task {task_id} is incomplete after {time_elapsed/60} minutes.")

    view_logger.error(f"Error - Task was unable to complete on time.")
    try:
        delete_task(task_id)
    except TaskQueue.DoesNotExist:
        view_logger.error(f"Unable to delete task {task_id}")
    print(1)
    return JsonResponse({'status': 'error',
                         'data': 'Task was unable to complete on time.'}, status=500)


# class QueueView(APIView):
#     @staticmethod
#     def get(request):
#         print("Inside get_queue_position", cache.get('tasks_in_queue'))
#         return Response({'queuePosition': cache.get('tasks_in_queue')}, status=200)


# @method_decorator(cache_page(60 * 60), name='dispatch')
class RecommendationsView(APIView):
    @staticmethod
    def get(request):
        username = request.query_params.get('username')
        site = request.query_params.get('site')
        if not username:
            return Response({"error": "Username is required"}, status=400)

        task = get_user_recs_task.delay(username, site)
        TaskQueue.objects.create(task_id=task.id)
        # task_position = len(TaskQueue.objects.all())

        UsernameCache.objects.get_or_create(username=username)
        print(f"Returning response with taskId of {task.id} from RecommendationsView")
        return Response({'taskId': task.id},
                         # 'queuePosition': task_position},
                        status=202)


# @method_decorator(cache_page(60 * 60), name='dispatch')
class AffinityFinderView(APIView):
    @staticmethod
    def get(request):
        username = request.query_params.get('username')
        site = request.query_params.get('site')
        if not username:
            return Response("Username is required", status=400)

        task = get_user_affs_task.delay(username, site)
        TaskQueue.objects.create(task_id=task.id)

        UsernameCache.objects.get_or_create(username=username)
        print(f"Returning response with taskId of {task.id} from AffinityFinderView")
        return Response({'taskId': task.id},
                        # 'queuePosition': task_position},
                        status=202)


# @method_decorator(cache_page(60 * 60), name='dispatch')
class SeasonalStatsView(APIView):
    @staticmethod
    def get(request):
        username = request.query_params.get('username')
        site = request.query_params.get('site')
        if not username:
            return Response("Username is required", status=400)

        task = get_user_seasonal_stats_task.delay(username, site)
        TaskQueue.objects.create(task_id=task.id)
        # task_position = len(TaskQueue.objects.all())

        UsernameCache.objects.get_or_create(username=username)
        print(f"Returning response with taskId of {task.id} from SeasonalStatsView")
        print("Inside seasonal_stats_view", cache.get('tasks_in_queue'))
        return Response({'taskId': task.id},
                        # 'queuePosition': task_position},
                        status=202)


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
    test = request.GET.get('test', '[]')
    print("Inside get_queue_position", test)
    cache_info = cache.client.get_client().connection_pool.connection_kwargs
    print("Cache configuration (get_queue_position):", cache_info)
    print("Inside get_queue_position", cache.get('tasks_in_queue', 1))
    # view_logger.error("Inside get_queue_position", cache.get('tasks_in_queue', 0))
    return Response({'queuePosition': cache.get('tasks_in_queue', 1)}, status=200)


@api_view(('GET',))
@renderer_classes((JSONRenderer,))
def get_anime_img_urls(request):
    show_names = request.GET.get('show_names', '[]')
    print(show_names)
    view_logger.info(show_names)
    try:
        show_names_list = json.loads(show_names)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON format'}, status=400)
    # show_names_list = show_names.split(',')
    print(show_names_list)
    print(len(show_names_list))
    try:
        preserve_order = Case(*[When(name=name, then=pos) for pos, name in enumerate(show_names_list)])
        # anime_data = AnimeData.objects.filter(name__in=show_names_list).annotate(ordering=preserve_order).order_by('ordering').values('image_url', 'name')
        anime_data = [AnimeDataUpdated.objects.get(name=show) for show in show_names_list]

        image_urls = [data.image_url for data in anime_data]

        if len(image_urls) != len(show_names_list):
            print(f"length of image_urls : {len(image_urls)}")
            # Find the missing one and replace it with a placeholder empty string
            anime_names_with_images = [data['name'] for data in anime_data]
            image_urls = [data['image_url'] if data['name'] in anime_names_with_images else ''
                          for data in anime_data]

        return Response(image_urls)
    except AnimeDataUpdated.DoesNotExist:
        return Response("One or more of the shows requested does not exist.", status=400)


# def username_cache_view(request):
#     username = request.GET.get('username')
#     user_list = UsernameCache.objects.values_list('username', flat=True)
#     print("Current usernames : ",user_list)
#     print("Username to find : ", username)
#     if username in user_list:
#         return JsonResponse({'UserFound': True})
#     else:
#         # UsernameCache.objects.create(username=username)
#         return JsonResponse({'UserFound': False})


@require_POST
@csrf_exempt
def update_image_url_view(request):
    data = json.loads(request.body)
    print(data)
    show_name = data.get('show_name')
    print(show_name)
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




# def index(request, id):
#     # def get_items_from_todolist():
#     #     return [item.text for item in ls.item_set.all()]
#     ls = ToDoList.objects.get(id=id)
#     items = [item.text for item in ls.item_set.all()]
#     # return HttpResponse(f"<h2> Testing 1 2 3, items are : {items} </h2>")
#     return render(request, "main/base.html", {"ls": ls})
#
#
# def list(request,id):
#     ls = ToDoList.objects.get(id=id)
#     return render(request, "main/list.html", {"ls" : ls})


# def get_recommendations_og(request):
#     if request.method == "POST":
#         username = request.POST.get('name')
#         current_dir = Path(__file__).parent
#         # models_path = current_dir.parent / "models"
#         model = Model(model_filename=current_dir / "models" / "T4-1-50-RSDDP.h5")
#         errors, recommendations = model.predict_scores(username, db_type=1)
#         # Get your recommendations using your function (replace 'your_function' with the appropriate name)
#         # recommendations = your_function(username)
#         # recommendations = "test"
#         return render(request, 'main/recs.html', {'recommendations': recommendations})
#     else:
#         form = CreateNewList()
#     return render(request, 'main/create.html', {"form": form})


# class RecommendationsView2(APIView):
#     def get_recommendations(self, request):
#         username = request.POST.get('name')
#         # current_dir = Path(__file__).parent
#         # model = Model(model_filename=current_dir / "MLmodels" / current_model_name)
#         errors, recommendations = model.predict_scores(username, db_type=1)
#         return Response(errors)




# def create(request):
#     if request.method == 'POST':
#         form = CreateNewList(request.POST)
#         if form.is_valid():
#             n = form.cleaned_data["name"]
#             t = ToDoList(name=n)
#             t.save()
#             return HttpResponseRedirect(f"/{t.id}")
#     else:
#         form = CreateNewList()
#         return HttpResponseRedirect("Invalid input")
#     return render(request, "main/create.html", {"form": form})
#
#
# def home(request):
#     return render(request, "main/home.html", {"name" : "test"})
#
#
# def index2(request):
#     return HttpResponse("<h1> WA HA HA </h1>")
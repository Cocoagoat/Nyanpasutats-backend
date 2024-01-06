from .modules.Model import Model
from pathlib import Path
from rest_framework.views import APIView
from rest_framework.response import Response
from .modules.filenames import current_model_name
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator
from .modules.AffinityFinder import find_max_affinity
from .modules.Errors import UserListFetchError
from .modules.SeasonalStats import SeasonalStats

# class MyDataView(APIView):
#     def get(self, request, format=None):
#         data = your_python_script_function()  # This returns a list or dictionary
#         return Response(data)

# current_dir = Path(__file__).parent
# model = Model(model_filename=current_dir / "MLmodels" / current_model_name)


@method_decorator(cache_page(60 * 60), name='dispatch')
class RecommendationsView(APIView):
    @staticmethod
    def get(request):
        username = request.query_params.get('username')
        if not username:
            return Response({"error": "Username is required"}, status=400)

        # current_dir = Path(__file__).parent
        # model = Model(model_filename=current_dir / "MLmodels" / current_model_name)
        try:
            errors, recommendations = model.predict_scores(username, db_type=1)
        except UserListFetchError as e:
            return Response(e.message, status=e.status)

        return Response({"Errors": errors, "Recommendations": recommendations})


@method_decorator(cache_page(60 * 60), name='dispatch')
class SeasonalStatsView(APIView):
    @staticmethod
    def get(request):
        username = request.query_params.get('username')
        if not username:
            return Response("Username is required", status=400)
        try:
            seasonal_dict = SeasonalStats.get_user_seasonal_stats(username)
        except UserListFetchError as e:
            return Response(e.message, e.status)
        return Response(seasonal_dict)


@method_decorator(cache_page(60 * 60), name='dispatch')
class AffinityFinderView(APIView):
    @staticmethod
    def get(request):
        username = request.query_params.get('username')
        if not username:
            return Response("Username is required", status=400)
        try:
            affinities = find_max_affinity(username)
        except UserListFetchError as e:
            return Response(e.message, e.status)
        return Response(affinities)
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
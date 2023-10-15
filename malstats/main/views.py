from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import ToDoList, Item
from .forms import CreateNewList
from .modules.Model import Model
from pathlib import Path
# Create your views here.


def index(request, id):
    # def get_items_from_todolist():
    #     return [item.text for item in ls.item_set.all()]
    ls = ToDoList.objects.get(id=id)
    items = [item.text for item in ls.item_set.all()]
    # return HttpResponse(f"<h2> Testing 1 2 3, items are : {items} </h2>")
    return render(request, "main/base.html", {"ls": ls})


def list(request,id):
    ls = ToDoList.objects.get(id=id)
    return render(request, "main/list.html", {"ls" : ls})


def get_recommendations(request):
    if request.method == "POST":
        username = request.POST.get('username')
        current_dir = Path(__file__).parent
        # models_path = current_dir.parent / "models"
        model = Model(current_dir / "models" / "T4-1-50-RSDDP.h5")
        recommendations = model.predict_scores(username, db_type=1)
        # Get your recommendations using your function (replace 'your_function' with the appropriate name)
        # recommendations = your_function(username)
        # recommendations = "test"
        return render(request, 'main/recs.html', {'recommendations': recommendations})
    else:
        form = CreateNewList()
    return render(request, 'main/create.html', {"form": form})


def create(request):
    if request.method == 'POST':
        form = CreateNewList(request.POST)
        if form.is_valid():
            n = form.cleaned_data["name"]
            t = ToDoList(name=n)
            t.save()
            return HttpResponseRedirect(f"/{t.id}")
    else:
        form = CreateNewList()
        return HttpResponseRedirect("Invalid input")
    return render(request, "main/create.html", {"form": form})


def home(request):
    return render(request, "main/home.html", {"name" : "test"})


def index2(request):
    return HttpResponse("<h1> WA HA HA </h1>")
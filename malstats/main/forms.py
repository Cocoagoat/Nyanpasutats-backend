from django import forms


# This is a form with a single name field and a check button
class CreateNewList(forms.Form):
    name = forms.CharField(label="MAL Username", max_length=200)
    check = forms.BooleanField()
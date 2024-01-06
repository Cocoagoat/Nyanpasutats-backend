from django.test import TestCase, Client
from django.urls import reverse


class MyDataViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_post_request(self):
        response = self.client.post(reverse('recs'), {'username': 'BaronBrixius'})
        print(response.data)
        self.assertEqual(response.status_code, 200)
        # Add more assertions here as needed
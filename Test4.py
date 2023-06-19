import requests
import time
from general_utils import save_pickled_file
query = '''
query ($page: Int, $isadult : Boolean) {
  Page(page: $page, perPage: 100) {
    pageInfo {
      total
      perPage
      currentPage
      lastPage
      hasNextPage
    }
    media (type : ANIME, isAdult : $isadult) {
      tags {
        name
        category
      }
      genres
    }
  }
}
'''

url = "https://graphql.anilist.co"


def fetch_categories(page):
    print(f"Currently on page {page}")
    variables = {"page": page, "isadult" : False}
    response = requests.post(url, json={"query": query, "variables": variables})
    data = response.json()
    media_list = data["data"]["Page"]["media"]
    # categories = set()
    for media in media_list:
        for tag in media["tags"]:
            category_name = tag["category"]
            tag_name = tag["name"]
            if category_name not in unique_categories:
                categories_tags_dict[category_name]=[]
                unique_categories.append(category_name)
            if tag_name not in unique_tags:
                categories_tags_dict[category_name].append(tag_name)
                unique_tags.append(tag_name)
            # categories.add(tag["category"])
    return data["data"]["Page"]["pageInfo"]["hasNextPage"]


categories_tags_dict={}
unique_categories = []
unique_tags=[]
page = 1
has_next_page = True

while has_next_page:
    has_next_page = fetch_categories(page)
    # unique_categories.update(categories)
    time.sleep(1)
    page += 1
    # if page==40:
    #     break

setting_tags = [x for key,value in categories_tags_dict.items() for x in value if key.startswith("Setting")]
theme_tags = [x for key,value in categories_tags_dict.items() for x in value if key.startswith("Theme")]
cast_tags = [x for key,value in categories_tags_dict.items() for x in value if key.startswith("Cast")]

categories_tags_dict = {"Setting" : setting_tags, "Theme" : theme_tags, "Cast" : cast_tags,
                        "Demographic" : categories_tags_dict["Demographic"]}



save_pickled_file("categories.pickle",categories_tags_dict)
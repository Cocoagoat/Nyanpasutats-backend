try:
    import thread
except ImportError:
    import _thread as thread
import urllib3
from main.modules.AffinityFinder import find_max_affinity


if __name__ == '__main__':
    # start_monitoring(seconds_frozen=Sleep.LONG_SLEEP + 10, test_interval=100)
    # This monitors threads that have been dormant for more than the longest sleep settings
    # allowed. Should not happen in practice since we're sending all our API calls through
    # separate process with a timeout of 15 seconds.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    # ser = Service("C:\\Program Files (x86)\\chromedriver.exe")
    # op = webdriver.ChromeOptions()
    # op.add_argument("--headless")
    # op.add_argument("--lang=ja")
    # op.add_argument('--blink-settings=imagesEnabled=false')
    # driver = webdriver.Chrome(service=ser, options=op)
    #
    # BASE_PATH = "data"
    # HTML_PATH = BASE_PATH + "/html"
    # USER_PATH = BASE_PATH + "/users"
    find_max_affinity("Tiali")
    print("Some changes")
    print("Some more changes")


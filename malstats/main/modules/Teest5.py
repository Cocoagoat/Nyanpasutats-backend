from Test import main
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'animisc.settings')
import django
django.setup()

if __name__ == '__main__':
    print(5)
    main()

# Generated by Django 4.2.5 on 2024-06-06 15:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0009_animedataupdated_alter_animedata_managers_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='usernamecache',
            name='expires_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]

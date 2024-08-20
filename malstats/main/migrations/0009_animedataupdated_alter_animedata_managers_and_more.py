# Generated by Django 4.2.5 on 2024-05-15 05:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0008_alter_animedata_managers_alter_taskqueue_managers_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='AnimeDataUpdated',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('mal_id', models.IntegerField()),
                ('mean_score', models.DecimalField(decimal_places=2, max_digits=3)),
                ('scores', models.IntegerField()),
                ('members', models.IntegerField()),
                ('episodes', models.PositiveSmallIntegerField(null=True)),
                ('duration', models.DecimalField(decimal_places=2, max_digits=5, null=True)),
                ('type', models.CharField(max_length=10, null=True)),
                ('year', models.PositiveSmallIntegerField(null=True)),
                ('season', models.PositiveSmallIntegerField(null=True)),
                ('name', models.CharField(max_length=300)),
                ('image_url', models.CharField(max_length=200, null=True)),
            ],
        ),
        migrations.AlterModelManagers(
            name='animedata',
            managers=[
            ],
        ),
        migrations.AlterModelManagers(
            name='taskqueue',
            managers=[
            ],
        ),
        migrations.AlterModelManagers(
            name='usernamecache',
            managers=[
            ],
        ),
    ]
from django.db import models

# Create your models here.
class Member(models.Model):
    objects = models.Manager()
    user_id = models.CharField(max_length=50, primary_key=True)
    user_pw = models.CharField(max_length=50)
    user_name = models.CharField(max_length=50)
    c_date = models.DateTimeField()

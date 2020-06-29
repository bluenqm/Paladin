from django.contrib import admin

# Register your models here.
from server.models import Project, Label, Document, Annotation, Prediction, Profile, Proficiency

admin.site.register(Project)
admin.site.register(Label)
admin.site.register(Document)
admin.site.register(Annotation)
admin.site.register(Prediction)
admin.site.register(Profile)
admin.site.register(Proficiency)
from io import TextIOWrapper

import pandas
from django import forms
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.models import User
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views.generic import TemplateView, CreateView, ListView
from rest_framework.generics import get_object_or_404

from server.models import Project, Document, Annotation
from server.permissions import SuperUserMixin


class AnnotationView(LoginRequiredMixin, TemplateView):
    def get_template_names(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        return project.get_template_name()


class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ('name', 'description', 'users')


class ProjectsView(LoginRequiredMixin, CreateView):
    form_class = ProjectForm
    template_name = 'projects.html'

    # Redirect to 'projects' after the Project is successfully created
    def form_valid(self, form):
        super().form_valid(form)
        return HttpResponseRedirect(reverse('projects'))


class DatasetView(LoginRequiredMixin, ListView):
    template_name = 'admin/dataset.html'
    paginate_by = 10

    def get_queryset(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        return project.documents.all().filter(seed=False).order_by('id')

    def csv_to_documents(self, project, file):
        file_tsv = TextIOWrapper(file, encoding='utf-8')
        data = pandas.read_csv(file_tsv, header=0, delimiter="\t")
        trimmed_data = data#.head(n=50)

        documents = []

        for index, row in trimmed_data.iterrows():
            text = row["sentence"]
            documents.append(Document(text=text, project=project))

        return documents

    def post(self, request, *args, **kwargs):
        project = get_object_or_404(Project, pk=kwargs.get('project_id'))
        file = request.FILES['file'].file
        documents = self.csv_to_documents(project, file)

        Document.objects.bulk_create(documents)
        print("Done uploading documents!")
        # TODO: allocate new batch here
        project.update_predicted_scores()
        project.update_assigned_annotators()
        return HttpResponseRedirect(reverse('dataset', args=[project.id]))


class SeedDatasetView(LoginRequiredMixin, ListView):
    template_name = 'admin/seed_dataset.html'
    paginate_by = 10

    def get_queryset(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        data = project.documents.all().filter(seed=True).order_by('id')
        return data

    def post(self, request, *args, **kwargs):
        project = get_object_or_404(Project, pk=kwargs.get('project_id'))
        project.add_user(username='truth')
        file = request.FILES['file'].file
        df = project.csv_to_documents_annotations(file)
        project.init_classifier(df)  # initialize the classifier and do the first training using uploaded dataset
        return HttpResponseRedirect(reverse('seed', args=[project.id]))


class LabelView(SuperUserMixin, LoginRequiredMixin, TemplateView):
    template_name = 'admin/label_management.html'


class StatsView(SuperUserMixin, LoginRequiredMixin, TemplateView):
    template_name = 'admin/stats.html'


class SettingView(LoginRequiredMixin, TemplateView):
    template_name = 'admin/setting.html'


class IndexView(TemplateView):
    template_name = 'index.html'


class ProfileView(SuperUserMixin, LoginRequiredMixin, ListView):
    template_name = 'admin/profile.html'

    def get_queryset(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        project_user = project.users.all().exclude(username='truth')
        return project_user

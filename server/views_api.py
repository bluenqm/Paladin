from collections import Counter
from itertools import chain

from django.contrib.auth.models import User
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.generics import get_object_or_404
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from server.models import Project, Label, Document, Annotation, Prediction
from server.serializers import ProjectSerializer, LabelSerializer, DocumentSerializer, AnnotationSerializer, \
    PredictionSerializer, UserSerializer, ProfileSerializer


class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return self.request.user.projects

    @action(methods=['get'], detail=True)
    def progress(self, request, pk=None):
        project = self.get_object()
        return Response(project.get_progress())

    @action(methods=['get'], detail=True)
    def userstats(self, request, pk=None):
        project = self.get_object()
        return Response(project.userstats())


class LabelViewSet(viewsets.ModelViewSet):
    queryset = Label.objects.all()
    serializer_class = LabelSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return self.queryset.filter(project=self.kwargs['project_id'])

    def perform_create(self, serializer):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        serializer.save(project=project)


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return User.objects.all()

    def put(self, request, *args, **kwargs):
        for user_data in request.data:
            user = self.get_object(user_data['id'])
            print(user)


class ProfileViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = ProfileSerializer
    permission_classes = [IsAuthenticated]


class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    permission_classes = [IsAuthenticated]

    def init_learning(self):  # Proof of concept
        print('Called from init_learning')
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        project.start_new_batch()

    def get_queryset(self):
        user = self.request.user
        assigned_docs = Document.objects.filter(project=self.kwargs['project_id'], assigned_to=user)
        results = assigned_docs
        if not results:
            self.init_learning()
            results = Document.objects.filter(project=self.kwargs['project_id'], assigned_to=user)
        return results
        #  We can use filter to display only docs that assigned to the request user

    def perform_create(self, serializer):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        serializer.save(project=project)


class AnnotationViewSet(viewsets.ModelViewSet):  # return all annotation by user
    queryset = Annotation.objects.all()
    serializer_class = AnnotationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        return Annotation.objects.filter(user=user)


class AnnotationDocViewSet(viewsets.ModelViewSet):  # return all annotation by document
    queryset = Annotation.objects.all()
    serializer_class = AnnotationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Annotation.objects.filter(document=self.kwargs['document_id'])

    def perform_create(self, serializer):
        document = get_object_or_404(Document, pk=self.kwargs['document_id'])
        user = self.request.user
        serializer.save(document=document, user=user)

        # Update the doc as annotated
        document.annotated = True
        document.save()
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        project.current_step = project.current_step + 1
        project.save()

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        self.perform_destroy(instance)

        # check if doc is still annotated
        document = get_object_or_404(Document, pk=self.kwargs['document_id'])
        annotations = document.annotations.all()
        if not annotations:
            document.annotated = False
            document.save()

        return Response(data='delete success')


class PredictionViewSet(viewsets.ModelViewSet):
    queryset = Prediction.objects.all().order_by('prob')
    serializer_class = PredictionSerializer
    permission_classes = [IsAuthenticated]

'''This API View is used when the annotator click Finish Annotation (batch)
It will (1) collect annotations (2) update the classifier, and (3) allocate new documents for annotation'''
class ProactiveView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        user = self.request.user
        project.start_new_batch(user)
        return Response({"message": "Done Allocating New Batch"})


class ProjectStatsAPI(APIView):
    pagination_class = None
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        p = get_object_or_404(Project, pk=self.kwargs['project_id'])
        labels = [label.text for label in p.labels.all()]
        users = [user.username for user in p.users.all()]
        docs = [doc for doc in p.documents.all()]
        nested_labels = [[a.label.text for a in doc.get_annotations()] for doc in docs]
        nested_users = [[a.user.username for a in doc.get_annotations()] for doc in docs]

        label_count = Counter(chain(*nested_labels))
        label_data = [label_count[name] for name in labels]

        user_count = Counter(chain(*nested_users))
        user_data = [user_count[name] for name in users]

        response = {'label': {'labels': labels, 'data': label_data},
                    'user': {'users': users, 'data': user_data}}

        return Response(response)


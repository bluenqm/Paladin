from django.urls import path
from rest_framework import routers

from .views import ProjectsView, DatasetView, LabelView, StatsView, SettingView, ProfileView, IndexView, \
    AnnotationView, SeedDatasetView
from .views_api import ProjectViewSet, LabelViewSet, DocumentViewSet, PredictionViewSet, UserViewSet, \
    AnnotationDocViewSet, AnnotationViewSet, ProactiveView, ProjectStatsAPI

router = routers.DefaultRouter()
router.register(r'projects', ProjectViewSet)
router.register(r'projects/(?P<project_id>\d+)/labels', LabelViewSet)
router.register(r'projects/(?P<project_id>\d+)/docs', DocumentViewSet)
router.register(r'projects/(?P<project_id>\d+)/docs/(?P<document_id>\d+)/annotations', AnnotationDocViewSet)
router.register(r'projects/(?P<project_id>\d+)/annotations', AnnotationViewSet)
router.register(r'predictions', PredictionViewSet)
router.register(r'projects/(?P<project_id>\d+)/users', UserViewSet)
router.register(r'profiles', UserViewSet)

# The API URLs are now determined automatically by the router.
urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('projects/', ProjectsView.as_view(), name='projects'),
    path('projects/<int:project_id>/', AnnotationView.as_view(), name='annotation'),
    path('projects/<int:project_id>/docs/', DatasetView.as_view(), name='dataset'),
    path('projects/<int:project_id>/seeds/', SeedDatasetView.as_view(), name='seed'),
    path('projects/<int:project_id>/labels/', LabelView.as_view(), name='label-management'),
    path('projects/<int:project_id>/stats/', StatsView.as_view(), name='stats'),
    path('projects/<int:project_id>/setting/', SettingView.as_view(), name='setting'),
    path('projects/<int:project_id>/profiles/', ProfileView.as_view(), name='profile'),
    path('api/projects/<int:project_id>/proactive', ProactiveView.as_view(), name='proactive'),
    path('api/projects/<int:project_id>/stats/', ProjectStatsAPI.as_view(), name='stats-api'),
]


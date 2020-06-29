from django.contrib.auth.models import User
from django.db.migrations import serializer
from rest_framework import serializers

from server.models import Project, Label, Document, Annotation, Prediction, Profile


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ('id', 'name', 'description', 'guideline', 'created_at', 'updated_at', 'users',
                  'active_learning_strategy', 'sampling_threshold',
                  'proactive_learning_strategy', 'proficiency_threshold',
                  'allocate_new_batch',
                  'steps_before_retraining', 'current_step', 'samples_per_session')


class LabelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Label
        fields = ('id', 'text', 'shortcut', 'background_color', 'text_color')


class UserSerializer(serializers.ModelSerializer):
    user_role = serializers.SerializerMethodField()
    doc_per_session = serializers.SerializerMethodField()

    def get_user_role(self, instance):
        return instance.profile.user_role

    def get_doc_per_session(self, instance):
        return instance.profile.doc_per_session

    class Meta:
        model = User
        fields = ('id', 'username', 'user_role', 'doc_per_session')


class DocumentSerializer(serializers.ModelSerializer):
    annotations = serializers.SerializerMethodField()
    predictions = serializers.SerializerMethodField()

    def get_annotations(self, instance):
        project = instance.project
        model = project.get_annotation_class()
        serializer = project.get_annotation_serializer()
        annotations = model.objects.filter(document=instance.id)
        serializer = serializer(annotations, many=True)
        return serializer.data

    def get_predictions(self, instance):
        project = instance.project
        model = project.get_prediction_class()
        serializer = project.get_prediction_serializer()
        predictions = model.objects.filter(document=instance.id)
        serializer = serializer(predictions, many=True)
        return serializer.data

    def __init__(self, *args, **kwargs):
        many = kwargs.pop('many', True)
        super(DocumentSerializer, self).__init__(many=many, *args, **kwargs)

    class Meta:
        model = Document
        fields = ('id', 'text', 'annotated', 'approved', 'annotations', 'predictions', 'assigned_to')


class AnnotationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Annotation
        fields = ('id', 'label')


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ('id', 'label', 'prob')


class ProfileSerializer(serializers.ModelSerializer):
    user_role = serializers.SerializerMethodField()
    doc_per_session = serializers.SerializerMethodField()

    def get_user_role(self, instance):
        return instance.profile.user_role

    def get_doc_per_session(self, instance):
        return instance.profile.doc_per_session

    class Meta:
        model = User
        fields = ('id', 'username', 'user_role', 'doc_per_session')

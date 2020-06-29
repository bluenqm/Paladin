import random
from io import TextIOWrapper

import pandas
import random
from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.urls import reverse

from server.classifier import TextClassifier
from server.utils import get_key_choices

# 19 Feb: additional information of User, needed to distinguish 3 user types: Admin / Manager / Annotator
USER_ROLE_CHOICES = (
    ('admin', 'ADMIN'),
    ('truth', 'TRUTH'),
    ('manager', 'MANAGER'),
    ('annotator', 'ANNOTATOR'),
)

MAX_PROACTIVE_LEARNING_SCORE = 999
MAX_SEED_TO_ASSIGN = 20

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    user_role = models.CharField(max_length=10, choices=USER_ROLE_CHOICES, default='annotator')
    doc_per_session = models.IntegerField(default=1)


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()


class Project(models.Model):
    name = models.CharField(max_length=100)
    description = models.CharField(max_length=500)
    guideline = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    users = models.ManyToManyField(User, related_name='projects')

    active_learning_strategy = models.IntegerField(default=1)
    sampling_threshold = models.DecimalField(max_digits=3, decimal_places=1, default=0.5)
    proactive_learning_strategy = models.IntegerField(default=1)
    proficiency_threshold = models.DecimalField(max_digits=3, decimal_places=1, default=0.5)
    allocate_new_batch = models.IntegerField(default=1)
    steps_before_retraining = models.IntegerField(default=100)
    current_step = models.IntegerField(default=0)
    samples_per_session = models.IntegerField(default=100)
    classifier = None

    def get_progress(self):
        docs = self.get_annotated_documents()
        total = self.documents.count()
        remaining = docs.count()
        return {'total': total, 'remaining': remaining}

    def userstats(self):
        all_annotators = list(self.users.all().exclude(username='truth'))
        proficiencies_list = []
        for annotator in all_annotators:
            profs = annotator.proficiencies
            for prof in profs:
                proficiencies_list.append({'label': prof.label, 'prof': prof.prob})

        users = [user.username for user in all_annotators]
        return {'user': users, 'proficiencies': proficiencies_list}

    def get_documents(self):
        docs = self.documents.all()
        return docs

    def get_annotated_documents(self):
        docs = self.documents.all()
        docs = docs.filter(annotated=False)
        return docs

    def load_classifier(self):
        num_labels = len(self.labels.all())
        if num_labels == 0:
            return
        path_to_model = './' + str(self.id) + '/trained_model/'
        self.classifier = TextClassifier(num_labels=num_labels, path_to_model=path_to_model, force_retrain=False)

    def init_classifier_from_approved_docs(self):
        num_labels = len(self.labels.all())
        path_to_model = './' + str(self.id) + '/trained_model/'
        assert (num_labels > 0)
        # Create training df base on approved documents
        train_df = self.get_training_data_from_approved_docs()
        self.classifier = TextClassifier(num_labels=num_labels, train_df=train_df, path_to_model=path_to_model)

    def get_training_data_from_approved_docs(self):
        docs = self.documents.all().filter(approved=True)
        labels = Label.objects.all()
        truth_annotator = User.objects.filter(username='truth').first()

        column_names = ["text", "labels"]
        df = pandas.DataFrame(columns=column_names)

        # Get labels from truth_annotator
        for doc in docs:
            label_list = []
            for label in labels:
                annotations = Annotation.objects.filter(label=label, document=doc, user=truth_annotator)
                if annotations:
                    label_list.append(1)
                else:
                    label_list.append(0)
            #extracted_labels = zip(label_list)
            df_row = pandas.DataFrame({'text': doc.text, 'labels': [label_list]})
            df = pandas.concat([df_row, df], ignore_index=True)

        return df



    def init_classifier(self, train_df):
        num_labels = len(self.labels.all())
        path_to_model = './models/' + str(self.id) + '/trained_model/'
        assert(num_labels > 0)
        self.classifier = TextClassifier(num_labels=num_labels, train_df=train_df, path_to_model=path_to_model)

    def get_annotation_class(self):
        return Annotation

    def get_prediction_class(self):
        return Prediction

    def get_annotation_serializer(self):
        from .serializers import AnnotationSerializer
        return AnnotationSerializer

    def get_prediction_serializer(self):
        from .serializers import PredictionSerializer
        return PredictionSerializer

    def get_absolute_url(self):
        return reverse('annotation', args=[self.id])

    # Add an existing user to the project
    def add_user(self, username):
        user = User.objects.all().filter(username=username)
        users = self.users.all()
        if users.filter(username=username).exists():
            return
        users |= user
        self.users.set(users)
        self.save()

    # Todo: Need testing
    # If there is only one annotator, we should approve all
    def approve_annotations(self, user, document):
        predictions = document.predictions.filter(prob__gte=0.5)
        annotations = document.annotations.filter(user=user)
        truth_annotators = User.objects.filter(username='truth')
        truth_annotator = truth_annotators[0]
        users = self.users.all()

        # If current annotator agrees with the prediction on every label → approve
        if (len(users) <= 2) or (list(predictions) == list(annotations)):
            print('We should approve')
            for annotation in annotations:
                truth_annotation = Annotation(label=annotation.label, document=document, user=truth_annotator)
                truth_annotation.save()
        # If current annotator NOT agrees with the prediction on every label
        else:
            annotations_from_other_user = document.annotations.exclude(user=user)
            # If we have another annotation → approve
            if len(annotations_from_other_user) > 0:
                print('We should approve')
                labels = Label.objects.all()
                for label in labels:
                    yes_count = annotations.filter(label=label) + annotations_from_other_user.filter(label=label)
                    if yes_count >= 2:  # annotators agree
                        truth_annotation = Annotation(label=label, document=document, user=truth_annotator)
                        truth_annotation.save()
                    else:  # annotators not agree
                        pred = predictions.filter(label=label)
                        if pred.prob >= 0.5:
                            truth_annotation = Annotation(label=label, document=document, user=truth_annotator)
                            truth_annotation.save()
            else:
                print('We should NOT approve')

    def train_model(self):
        if (self.classifier == None):
            self.init_classifier_from_approved_docs()

    # TODO: need testing
    '''Run the classifier on all unapproved docs to get the predict scores
    Require a MODEL, else will train a MODEL on approved docs'''
    def update_predicted_scores(self):
        print('Getting predicted scores')  # classifier != null

        # Get classifier here
        if self.classifier == None:
            self.load_classifier()

        # Get all unapproved docs, apply the classifier to get prediction scores
        docs = self.documents.all().filter(approved=False)
        list_doc = []
        for doc in docs:
            list_doc.append(doc.text)
        raw_outputs = self.classifier.classify(list_doc)
        labels = self.labels.all()
        for i in range(len(raw_outputs)):  # len(raw_outputs) == len(docs)
            assert(len(raw_outputs) == len(docs))
            doci = docs[i]
            outputi = raw_outputs[i]  # len(outputi) == len(labels)
            assert(len(outputi) == len(labels))
            for j in range(len(outputi)):
                label = labels[j]
                defaults = {'prob': outputi[j]}
                try:
                    obj = Prediction.objects.get(document=doci, label=label)
                    for key, value in defaults.items():
                        setattr(obj, key, value)
                    obj.save()
                except Prediction.DoesNotExist:
                    new_values = {'document': doci, 'label': label}
                    new_values.update(defaults)
                    obj = Prediction(**new_values)
                    obj.save()

    # TODO: need testing
    ''''''
    def update_assigned_annotators(self):
        # Update active learning scores, get docs ordered by active learning scores --> which doc need annotation?
        docs = self.documents.all().filter(approved=False, seed=False)
        for i in range(len(docs)):
            preds = docs[i].predictions.all().filter(prob__gte=self.sampling_threshold)
            active_learning_score = 0
            for j in range(len(preds)):
                active_learning_score = active_learning_score + preds[j].label.count * preds[j].prob
            docs[i].active_learning_score = active_learning_score
        docs = self.documents.all().filter(approved=False, seed=False).order_by('active_learning_score')

        # Get annotators that need new doc allocation --> which annotators need reassign?
        all_annotators = list(self.users.all().exclude(username='truth'))
        annotators = [ann for ann in all_annotators if (ann.profile.doc_per_session > 0 and not Document.objects.filter(
            assigned_to__username=ann.username))]
        print('There are ' + str(len(annotators)) + ' annotators in this project need reassign!')

        # Allocate unapproved docs
        for i in range(len(docs)):  # for each document, update the assigned annotator
            if len(annotators) == 0:  # no available annotator
                docs[i].assigned_to.clear()
            else:
                preds = docs[i].predictions.all().filter(prob__gte=self.sampling_threshold)
                anns = docs[i].annotations.all()
                assigned_annotator = 0
                proactive_learning_score_min = self.calculate_proactive_learning_score(annotators[0], preds, anns)
                for k in range(1, len(annotators)):
                    proactive_learning_score = self.calculate_proactive_learning_score(annotators[k], preds, anns)
                    if proactive_learning_score < proactive_learning_score_min:
                        proactive_learning_score_min = proactive_learning_score
                        assigned_annotator = k
                if proactive_learning_score_min == MAX_PROACTIVE_LEARNING_SCORE:  # can't find any annotator
                    docs[i].assigned_to.clear()
                else:
                    docs[i].assigned_to.clear()
                    docs[i].assigned_to.add(annotators[assigned_annotator])
                    annotators[assigned_annotator].profile.doc_per_session = annotators[
                                                                                 assigned_annotator].profile.doc_per_session - 1
                    if annotators[assigned_annotator].profile.doc_per_session == 0:
                        annotators.pop(assigned_annotator)
            docs[i].save()

    def calculate_proactive_learning_score(self, annotator, predictions, annotations):
        score = 0
        annotated_by_this_user = False
        for ann in annotations:
            if ann.user == annotator:
                annotated_by_this_user = True
                break
        if annotated_by_this_user:
            return MAX_PROACTIVE_LEARNING_SCORE
        proficiencies = Proficiency.objects.all().filter(annotator=annotator)
        if not proficiencies:
            labels = self.labels.all()
            for j in range(len(labels)):
                new_pro = Proficiency()
                new_pro.annotator = annotator
                new_pro.label = labels[j]
                new_pro.prob = random.uniform(0, 1)
                new_pro.save()
            proficiencies = Proficiency.objects.all().filter(annotator=annotator)
        for i in range(len(predictions)):
            for j in range(len(proficiencies)):
                if predictions[i].label == proficiencies[j].label:
                    score = score + predictions[i].prob * proficiencies[j].prob
        return score

    def start_new_batch(self, user):
        # Approve annotations and Remove those docs assigned to this user
        assigned_docs = Document.objects.filter(assigned_to__username=user.username)
        for i in range(0, len(assigned_docs)):
            self.approve_annotations(user, assigned_docs[i])
            assigned_docs[i].assigned_to.remove(user)
            assigned_docs[i].save()

        # Start new batch
        self.update_predicted_scores()
        self.update_assigned_annotators()

    def get_template_name(self):
        template_name = 'annotation.html'
        return template_name

    def __str__(self):
        return self.name

    def csv_to_documents_annotations(self, file):
        all_user = self.users.all().exclude(username='truth')
        truth_user = self.users.all().filter(username='truth').first()
        file_tsv = TextIOWrapper(file, encoding='utf-8')
        data = pandas.read_csv(file_tsv, header=0, delimiter="\t", encoding='latin1')
        trimmed_data = data#.head(n=10)
        labels = self.labels.all().order_by('id')
        if len(labels) == 0:  # TODO: have not defined labels --> use labels in seed file
            color_list = ['#0351C1', '#7EB3FF', '#0E73B7', '#45D09E', '#116315', '#A7E541', '#FFD600', '#FE9E76',
                          '#01142F', '#64C7FF', '#6050A7', '#00848C', '#1E3C00', '#8CBA51', '#D6C21A', '#F85C50',
                          '#4A69FF', '#C5CCDF', '#1EC9E8', '#004156', '#35D073', '#748700', '#D2AA1B', '#FFA96B',]
            i = 0
            for col in data.columns:
                if col != 'sentence':
                    c1 = color_list[i]
                    i = i + 1
                    if i >= len(color_list):
                        i = 0
                    new_label = Label(text=col, project=self, background_color=c1)
                    new_label.save()
        min_label_row_length = min(len(labels), len(trimmed_data.columns) - 1)

        for index, row in trimmed_data.iterrows():
            text = row["sentence"]
            document = Document(text=text, project=self, annotated=True, approved=True, seed=True)
            document.save()
            # Assign seed data to every user unless the seed data is too big
            if len(trimmed_data.index) <= MAX_SEED_TO_ASSIGN:
                document.assigned_to.set(all_user)
            document.save()
            # Put annotations as truth
            for j in range(0, min_label_row_length):
                label = row[j+1]
                if label == 1:
                    ann = Annotation(user=truth_user, document=document, label=labels[j])
                    ann.save()
        return data


class Label(models.Model):
    KEY_CHOICES = get_key_choices()

    text = models.CharField(max_length=100)
    shortcut = models.CharField(max_length=15, blank=True, null=True, choices=KEY_CHOICES)
    project = models.ForeignKey(Project, related_name='labels', on_delete=models.CASCADE)
    background_color = models.CharField(max_length=7, default='#209cee')
    text_color = models.CharField(max_length=7, default='#ffffff')
    count = models.IntegerField(default=0)

    def __str__(self):
        return self.text

    class Meta:
        unique_together = (
            ('project', 'text'),
            ('project', 'shortcut')
        )


class Document(models.Model):
    text = models.TextField()
    project = models.ForeignKey(Project, related_name='documents', on_delete=models.CASCADE)
    annotated = models.BooleanField(default=False)
    approved = models.BooleanField(default=False)
    seed = models.BooleanField(default=False)
    assigned_to = models.ManyToManyField(User, related_name='documents', default=None, blank=True)
    active_learning_score = models.FloatField(default=0.0)

    def __str__(self):
        return self.text[:50]

    def get_annotations(self):
        annotations = self.annotations.all()
        return annotations


class Annotation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, related_name='annotations', on_delete=models.CASCADE)
    label = models.ForeignKey(Label, on_delete=models.CASCADE)

    class Meta:
        unique_together = ('document', 'user', 'label')


class Prediction(models.Model):
    document = models.ForeignKey(Document, related_name='predictions', on_delete=models.CASCADE)
    label = models.ForeignKey(Label, on_delete=models.CASCADE)
    prob = models.FloatField(default=0.0)

    class Meta:
        unique_together = ('document', 'label')


class Proficiency(models.Model):
    annotator = models.ForeignKey(User, related_name='proficiencies', on_delete=models.CASCADE)
    label = models.ForeignKey(Label, on_delete=models.CASCADE)
    prob = models.FloatField(default=0.5)

    class Meta:
        unique_together = ('annotator', 'label')

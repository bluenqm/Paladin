import pandas as pd
import torch
from simpletransformers.classification import MultiLabelClassificationModel
import os.path
from sklearn.metrics import precision_recall_fscore_support as score
import logging
import gc

logging.basicConfig(filename='app.log', filemode='w', format='%(message)s')

class TextClassifier():
    model = None
    num_labels = 2
    num_train_epochs = 5
    path_to_model = './path_to_model/'

    def process_data_frame(self, df):
        print('Processing data frame ... ')
        if 'id' in df.columns:
            df = df.drop(['id'], axis=1)
        assert len(df.columns) >= 2
        thislist = [df.iloc[:, 1].tolist()]
        for i in range(2, len(df.columns)):
            current_list = df.iloc[:, i].tolist()
            thislist.append(current_list)
        df['labels'] = list(zip(*thislist))
        df['text'] = df['sentence'].apply(lambda x: x.replace('\n', ' '))
        df = df[['text', 'labels']]
        print(df.head())
        print('Processing data frame Finished!')
        return df

    def __init__(self, num_labels, train_df, path_to_model='./path_to_model/', force_retrain=False, num_train_epochs = 5):
        print('Initializing ... ')
        self.num_labels = num_labels
        self.path_to_model = path_to_model
        self.num_train_epochs = num_train_epochs

        if (force_retrain) or (not os.path.exists(path_to_model + 'config.json')):
            train_df = self.process_data_frame(train_df)
            print(train_df.head())

            self.model = MultiLabelClassificationModel(model_type='bert', model_name='bert-base-cased',
                                                       num_labels=self.num_labels,
                                                       args={'output_dir': self.path_to_model,
                                                             'reprocess_input_data': True,
                                                             'overwrite_output_dir': True,
                                                             'num_train_epochs': self.num_train_epochs})
            self.train(train_df)
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

        self.model = MultiLabelClassificationModel(model_type='bert', model_name=self.path_to_model,
                                                   num_labels=self.num_labels,
                                                   args={'output_dir': self.path_to_model, 'reprocess_input_data': True,
                                                         'overwrite_output_dir': True,
                                                         'num_train_epochs': self.num_train_epochs})
        print('Initializing Finished!')

    def train(self, train_df):
        print('Training ... ')
        self.model.train_model(train_df)
        print('Training Finished!')

    def evaluate(self, dev_df):
        print('Evaluating ... ')
        if 'id' in dev_df.columns:
            dev_df = dev_df.drop(['id'], axis=1)
        dev_df2 = self.process_data_frame(dev_df)
        result, model_outputs, wrong_predictions = self.model.eval_model(dev_df2)

        column_names = []
        for col in dev_df.columns:
            if (col != 'text' and col != 'labels' and col != 'sentence'):
                column_names.append(col + '_predicted')
        print(len(column_names))

        model_outputs = model_outputs.round(0)
        output_df = pd.DataFrame(model_outputs, columns=column_names)

        res_pd = pd.concat([dev_df, output_df], axis=1)
        file_name = '/home/minh/out.tsv'
        res_pd.to_csv(file_name, sep='\t', index=False)

        for col in column_names:
            print(col)
            predicted_column = res_pd.loc[:, col]
            predicted = predicted_column.values
            col2 = col.replace('_predicted', '')
            test_column = res_pd.loc[:, col2]
            y_test = test_column.values
            precision, recall, fscore, support = score(y_test, predicted)
            logging.warning('{}\t{:.2%}\t{:.2%}\t{:.2%}\t{}'.format(col[:4], precision[-1], recall[-1], fscore[-1], support[-1]))

        print('Evaluating Finished!')
        return result, model_outputs, wrong_predictions

    def classify(self, sentences):
        print('Predicting ... ')
        predictions, raw_outputs = self.model.predict(sentences)
        print(predictions)
        print(raw_outputs)
        print('Predicting Finished!')
        return raw_outputs

    def __del__(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

# body of destructor


# file_name = '/home/minh/Downloads/train.csv'
# df = pandas.read_csv(file_name, header=0, delimiter=",", encoding='latin1')
# print(df.head())
#
# df = df.head(1000)
# df = df.drop("id", axis=1)
#
# tc = TextClassifier(num_labels=6, train_df=df)
# sentences = ['text 1', 'text 2']
# tc.classify(sentences)


# model = MultiLabelClassificationModel(model_type='roberta', model_name='../16/trained_model/',
#                                                    num_labels=3,
#                                                    args={'output_dir': '../16/trained_model/', 'reprocess_input_data': True,
#                                                          'overwrite_output_dir': True,
#                                                          'num_train_epochs': 5})
# sentences = ['text 1', 'text 2']
# predictions, raw_outputs = model.predict(sentences)
# print(predictions)

train_file_name = '/home/minh/Downloads/jigsaw-toxic-comment-classification-challenge/kgpool.csv'
train_df = pd.read_csv(train_file_name, header=0, delimiter="\t", encoding='utf-8')

dev_file_name = '/home/minh/Downloads/jigsaw-toxic-comment-classification-challenge/kgtest.csv'
dev_df = pd.read_csv(dev_file_name, header=0, delimiter="\t", encoding='utf-8')
#print(df.head())

# Use this when train new dataframe
instances_per_iteration = 1000
train_df = train_df.sample(frac=1)
current_df = train_df.iloc[:instances_per_iteration, :]
train_df = train_df.iloc[instances_per_iteration:, :]

for i in range(1, 10):
    print('=====')
    print(i)
    logging.info(i)
    print(current_df.shape)
    logging.warning(current_df.shape)
    tc = TextClassifier(num_labels=6, train_df=current_df, path_to_model='/home/minh/Models/KG/', force_retrain=True, num_train_epochs = 3)
    result, model_outputs, wrong_predictions = tc.evaluate(dev_df)
    print(model_outputs)
    logging.info(model_outputs)
    del tc
    gc.collect()
    torch.cuda.empty_cache()

    df_for_train = train_df.iloc[:instances_per_iteration, :]
    train_df = train_df.iloc[instances_per_iteration:, :]
    current_df = pd.concat([current_df, df_for_train])

# Use this to load trained model
# model = MultiLabelClassificationModel(model_type='roberta', model_name='/home/minh/Models/ST/',
#                                                    num_labels=8,
#                                                    args={'output_dir': '/home/minh/Models/ST/', 'reprocess_input_data': True,
#                                                          'overwrite_output_dir': True,
#                                                          'num_train_epochs': 5})

# sentences = ['Piss Off Suck my dick you pussy', 'Go fuck yourself you stupid cunt 211.29.171.149', 'fucking getting none, I hope you choke, There\'s a bed made in Milltown just for you']
# tc.classify(sentences)

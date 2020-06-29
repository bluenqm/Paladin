import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel
import os.path
from sklearn.metrics import precision_recall_fscore_support as score


class TextClassifier():
    model = None
    num_labels = 2
    num_train_epochs = 3
    path_to_model = './path_to_model/'
    pos_weight = [1] * num_labels

    def process_data_frame(self, df):
        print('Processing data frame ... ')
        assert len(df.columns) >= 2
        thislist = [df.iloc[:, 1].tolist()]
        for i in range(2, len(df.columns)):
            current_list = df.iloc[:, i].tolist()
            thislist.append(current_list)
        df['labels'] = list(zip(*thislist))
        df['text'] = df['sentence'].apply(lambda x: x.replace('\n', ' '))
        df = df[['text', 'labels']]
        print('Processing data frame Finished!')
        return df

    def __init__(self, num_labels, train_df=None, pos_weight=[], path_to_model='./path_to_model/', force_retrain=True,
                 num_train_epochs=3):
        print('Initializing ... ')
        self.num_labels = num_labels
        self.path_to_model = path_to_model
        self.num_train_epochs = num_train_epochs
        if pos_weight:
            self.pos_weight = pos_weight

        if (force_retrain) or (not os.path.exists(path_to_model + 'config.json')):
            train_df = self.process_data_frame(train_df)

            self.model = MultiLabelClassificationModel(model_type='bert', model_name='bert-base-cased',
                                                       num_labels=self.num_labels,
                                                       args={'output_dir': self.path_to_model,
                                                             'reprocess_input_data': True,
                                                             'overwrite_output_dir': True,
                                                             'pos_weight': self.pos_weight,
                                                             'num_train_epochs': self.num_train_epochs})
            self.train(train_df)

        self.model = MultiLabelClassificationModel(model_type='bert', model_name=self.path_to_model,
                                                   num_labels=self.num_labels,
                                                   args={'output_dir': self.path_to_model,
                                                         'reprocess_input_data': True,
                                                         'overwrite_output_dir': True,
                                                         'pos_weight': self.pos_weight,
                                                         'num_train_epochs': self.num_train_epochs})
        print('Initializing Finished!')

    def train(self, train_df):
        print('Training ... ')
        self.model.train_model(train_df)
        print('Training Finished!')

    def evaluate(self, dev_df):
        print('Evaluating ... ')
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
            predicted_column = res_pd.loc[:, col]
            predicted = predicted_column.values
            col2 = col.replace('_predicted', '')
            test_column = res_pd.loc[:, col2]
            y_test = test_column.values
            precision, recall, fscore, support = score(y_test, predicted)
            print('{}\t{:.2%}\t{:.2%}\t{:.2%}\t{}'.format(col[:4], precision[-1], recall[-1], fscore[-1], support[-1]))

        print('Evaluating Finished!')
        return result, model_outputs, wrong_predictions

    def classify(self, sentences):
        print('Predicting ... ')
        predictions, raw_outputs = self.model.predict(sentences)
        print('Predicting Finished!')
        return raw_outputs

# For testing
# train_file_name = '/home/minh/Data/Processed/Client/train.tsv'
# train_df = pd.read_csv(train_file_name, header=0, delimiter="\t", encoding='utf-8')
# dev_file_name = '/home/minh/Data/Processed/Client/dev.tsv'
# dev_df = pd.read_csv(dev_file_name, header=0, delimiter="\t", encoding='utf-8')
# tc = TextClassifier(num_labels=15, train_df=train_df, path_to_model='/home/minh/Models/ST/', force_retrain=True,
#                     num_train_epochs=3)
# result, model_outputs, wrong_predictions = tc.evaluate(dev_df)
# print(model_outputs)
# sentences = ['Please cancel', 'Can I have an update', 'Can you resend the docs',
#              'Can you give the documents so I can sign?']
# raw_outputs = tc.classify(sentences)
# print(raw_outputs)

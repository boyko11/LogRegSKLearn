from service.data_service import DataService
from service.report_service import ReportService
from sklearn.linear_model import LogisticRegression


class Runner:

    def __init__(self, normalization_method='z'):
        self.normalization_method = normalization_method
        self.report_service = ReportService()

    def run(self):

        data = DataService.load_csv("data/wdbc.data")
        # column 1 is the id, column 2 is the label, the rest are features
        feature_data = data[:, 2:]
        labels = data[:, 1]
        normalized_data = DataService.normalize(data, method='z')
        normalized_feature_data = normalized_data[:, 2:]

        log_reg = LogisticRegression()
        log_reg_model = log_reg.fit(normalized_feature_data, labels)
        rounded_predictions = log_reg_model.predict(normalized_feature_data)
        predictions = log_reg_model.predict_proba(normalized_feature_data)

        print('Accuracy: {}'.format(log_reg.score(normalized_feature_data, labels)))

        record_ids = data[:, 0].flatten()
        self.report_service.report(predictions[:, 1], rounded_predictions, labels, record_ids)


if __name__ == "__main__":

    Runner(normalization_method='z').run()

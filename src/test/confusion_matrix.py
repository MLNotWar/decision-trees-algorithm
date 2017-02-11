import numpy as np


class ConfusionMatrix:
    def __init__(self, size=6):
        self.data = np.zeros((size, size))

    def update_data(self, predictions, expectations):
        for i in range(len(expectations)):
            for j in range(len(expectations[i])):
                self.data[expectations[i][j] - 1][predictions[i][j] - 1] += 1

    def accuracy(self):
        correct_predictions = 0
        incorrect_predictions = 0
        for i in range(len(self.data)):
            correct_predictions += self.data[i][i]
            incorrect_predictions += sum(self.data[i])
        return correct_predictions / incorrect_predictions

    def error_rate(self):
        return 1 - self.accuracy()

    def class_recall(self, class_num):
        if 0 > class_num >= len(self.data):
            return 0
        true_positive = self.data[class_num][class_num]
        true_found = sum(self.data[class_num])
        return true_positive / true_found if true_found else 0

    def unweighted_average_recall(self):
        total = 0
        for i in range(len(self.data)):
            total += self.class_recall(i)
        return total / len(self.data)

    def class_precision(self, class_num):
        if 0 > class_num >= len(self.data):
            return 0
        true_positive = self.data[class_num][class_num]
        true_expected = 0
        for i in range(len(self.data)):
            true_expected += self.data[i][class_num]
        return true_positive / true_expected if true_expected else 0

    def class_score(self, class_num, a=1):
        if 0 > class_num >= len(self.data):
            return 0
        recall = self.class_recall(class_num)
        precision = self.class_precision(class_num)
        return (1 + a ** 2) * precision * recall / (a ** 2 * precision + recall) if recall or precision else 0

    def generate_report(self):
        report = {
            "UAR": self.unweighted_average_recall(),
            "accuracy": self.accuracy(),
            "error_rate": self.error_rate(),
            "stats": []
        }
        for i in range(len(self.data)):
            report["stats"].append({
                "recall": self.class_recall(i),
                "precision": self.class_precision(i),
                "score": self.class_score(i)
            })
        return report

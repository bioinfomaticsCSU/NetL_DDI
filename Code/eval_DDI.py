
import copy
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


def construct_indicator(y_score, y):
    # rank the labels by the scores directly
    # num_label = np.sum(y, axis=1, dtype=np.int)
    # y_sort = np.fliplr(np.argsort(y_score, axis=1))
    # y_pred = np.zeros_like(y, dtype=np.int)
    # for i in range(y.shape[0]):
    #     for j in range(num_label[i]):
    #         y_pred[i, y_sort[i, j]] = 1
    y_pred = copy.copy(y_score)
    y_pred[y_pred > 0.47] = 1
    y_pred[y_pred <= 0.47] = 0
    return y_pred


def classifaction_report_csv(report, num_CV):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[1]
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        tnum = row_data[4].count(' ')
        if tnum > 0:
            row['f1_score'] = float(row_data[4][0:3])
            row['support'] = float(row_data[4][3 + tnum:])
        else:
            row['f1_score'] = float(row_data[4])
            row['support'] = float(row_data[5])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    filename = "classification_report_" + str(num_CV) + ".csv"
    dataframe.to_csv(filename, index = False)


def evaluate(pred, ture, num_CV):
    y_pred = construct_indicator(pred, ture)

    mi = f1_score(ture, y_pred, average="micro")
    ma = f1_score(ture, y_pred, average="macro")
    # w_f1 = f1_score(ture, y_pred, average="weighted")
    ma_pre = precision_score(ture, y_pred, average='macro')
    mi_pre = precision_score(ture, y_pred, average='micro')
    # w_pre = precision_score(ture, y_pred, average='weighted')
    ma_rea = recall_score(ture, y_pred, average='macro')
    mi_rea = recall_score(ture, y_pred, average='micro')
    # w_rea = recall_score(ture, y_pred, average='weighted')
    m_acc = accuracy_score(ture, y_pred)

    A = [i for i in range(86)]
    target_names = list(map(str, A))
    report = classification_report(ture, y_pred, target_names=target_names)
    classifaction_report_csv(report, num_CV)

    print('ma......', ma)
    print('ma_pre......', ma_pre)
    print('ma_rea......', ma_rea)
    print('mi......', mi)
    print('mi_pre......', mi_pre)
    print('mi_rea......', mi_rea)
    print('m_acc......', m_acc)

    return ma, ma_pre, ma_rea, mi, mi_pre, mi_rea, m_acc



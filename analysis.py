import torch
import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression

matplotlib.use("Agg")


def draw_loss(train_loss, val_loss, experimentname, filename='Loss.jpg', title='Loss of ', save=True):
    train_loss = np.mean(np.array(train_loss).reshape(-1, len(train_loss) // 200), axis=1)
    val_loss = np.mean(np.array(val_loss).reshape(-1, len(val_loss) // 200), axis=1)
    plt.figure(dpi=196)
    plt.plot(range(len(train_loss)), train_loss, c='g', label='Training')
    plt.plot(range(len(val_loss)), val_loss, c='b', label='Validation')
    plt.ylim(ymax=1)
    plt.ylim(ymin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title + experimentname)
    if save:
        plt.savefig(os.path.join('result', experimentname + filename))


def draw_acc(train_acc, val_acc, experimentname, filename='Accuracy.jpg', save=True):
    train_acc = np.mean(np.array(train_acc).reshape(-1, len(train_acc) // 200), axis=1)
    val_acc = np.mean(np.array(val_acc).reshape(-1, len(val_acc) // 200), axis=1)
    plt.figure(dpi=196)
    plt.plot(range(len(train_acc)), train_acc, c='g', label='Training')
    plt.plot(range(len(val_acc)), val_acc, c='b', label='Validation')
    plt.ylim(ymax=100)
    plt.ylim(ymin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of ' + experimentname)
    plt.legend()
    if save:
        plt.savefig(os.path.join('result', experimentname + filename))


def get_ILV_and_EFR(train_loss_list, val_loss_list, train_error_list, val_error_list, stride=1, section_len=20):
    train_loss_LV = []
    val_loss_LV = []
    train_acc_LV = []
    val_acc_LV = []

    train_loss_FR = []
    val_loss_FR = []
    train_acc_FR = []
    val_acc_FR = []
    x_list = np.linspace(1, section_len, section_len).reshape((-1, 1))

    for index in range((len(train_loss_list) - section_len) // stride + 1):
        train_loss_section = train_loss_list[index:(index + section_len)]
        regression = LinearRegression().fit(x_list, train_loss_section)
        train_loss_LV.append(regression.coef_[0])
        train_loss_FR.append(np.sum(np.maximum(np.diff(train_loss_section, n=1), 0)) / section_len)

    for index in range((len(val_loss_list) - section_len) // stride + 1):
        val_loss_section = val_loss_list[index:(index + section_len)]
        regression = LinearRegression().fit(x_list, val_loss_section)
        val_loss_LV.append(regression.coef_[0])
        val_loss_FR.append(np.sum(np.maximum(np.diff(val_loss_section, n=1), 0)) / section_len)

    q = pow(0.1, 2 / (len(train_loss_list)))
    # print("q",q)
    q_list = np.logspace(1, len(train_loss_LV), num=len(train_loss_LV), base=q)
    q_list = q_list / np.sum(q_list)
    # print("q_list",q_list[1:10])
    q_list2 = np.logspace(1, len(train_loss_LV), num=len(train_loss_LV), base=1 / q)
    q_list2 = q_list2 / np.sum(q_list2)
    # print("q_lis2",q_list2[1:10])
    train_loss_ILV = np.sum(train_loss_LV * q_list)
    train_loss_EFR = np.sum(train_loss_FR * q_list2)

    q = pow(0.1, 2 / (len(val_loss_list)))
    # print("q",q)
    q_list = np.logspace(1, len(val_loss_LV), num=len(val_loss_LV), base=q)
    q_list = q_list / np.sum(q_list)
    # print("q_list",q_list[1:10])
    q_list2 = np.logspace(1, len(val_loss_LV), num=len(val_loss_LV), base=1 / q)
    q_list2 = q_list2 / np.sum(q_list2)
    # print("q_lis2",q_list2[1:10])
    val_loss_ILV = np.sum(val_loss_LV * q_list)
    val_loss_EFR = np.sum(val_loss_FR * q_list2)

    for index in range((len(train_error_list) - section_len) // stride + 1):
        train_section = train_error_list[index:(index + section_len)]
        regression = LinearRegression().fit(x_list, train_section)
        train_acc_LV.append(-regression.coef_[0])
        train_acc_FR.append(np.sum(np.maximum(np.diff(train_section, n=1), 0)) / section_len)

    for index in range((len(val_error_list) - section_len) // stride + 1):
        val_section = val_error_list[index:(index + section_len)]
        regression = LinearRegression().fit(x_list, val_section)
        val_acc_LV.append(-regression.coef_[0])
        val_acc_FR.append(np.sum(np.maximum(np.diff(val_section, n=1), 0)) / section_len)

    q = pow(0.1, 2 / (len(train_acc_LV)))
    # print("q",q)
    q_list_acc = np.logspace(1, len(train_acc_LV), num=len(train_acc_LV), base=q)
    q_list_acc = q_list_acc / np.sum(q_list_acc)
    # print("q_list_acc",q_list_acc[1:10])
    q_list_acc2 = np.logspace(1, len(train_acc_LV), num=len(train_acc_LV), base=1 / q)
    q_list_acc2 = q_list_acc / np.sum(q_list_acc2)
    train_accuracy_ILV = np.sum(train_acc_LV * q_list_acc)
    train_accuracy_EFR = np.sum(train_acc_FR * q_list_acc2)

    q = pow(0.1, 2 / (len(val_acc_LV)))
    # print("q",q)
    q_list_acc = np.logspace(1, len(val_acc_LV), num=len(val_acc_LV), base=q)
    q_list_acc = q_list_acc / np.sum(q_list_acc)
    # print("q_list_acc",q_list_acc[1:10])
    q_list_acc2 = np.logspace(1, len(val_acc_LV), num=len(val_acc_LV), base=1 / q)
    q_list_acc2 = q_list_acc / np.sum(q_list_acc2)
    val_accuracy_ILV = np.sum(val_acc_LV * q_list_acc)
    val_accuracy_EFR = np.sum(val_acc_FR * q_list_acc2)

    ILV = {"train_loss": train_loss_ILV, "val_loss": val_loss_ILV, "train_accuracy": train_accuracy_ILV,
           "val_accuracy": val_accuracy_ILV}
    EFR = {"train_lloss": train_loss_EFR, "val_loss": val_loss_EFR, "train_accuracy": train_accuracy_EFR,
           "val_accuracy": val_accuracy_EFR}
    print("ILV", ILV, "EFR", EFR)
    return ILV, EFR


def get_overfitting_rate(train_acc, val_acc, train_loss, val_loss):
    begin_point = -10
    hypothesis_threshold = 0.01
    choice_list = train_acc[begin_point:-1]
    # print("choice_list",choice_list)
    train_acc_mean, train_acc_var = np.mean(choice_list), np.var(choice_list) * np.sqrt(len(choice_list))
    # print("train_acc_mean",train_acc_mean,"train_acc_var",train_acc_var)
    index = begin_point
    hypothesis_prob = 1
    while -index < len(train_acc):
        index -= 1
        testing_cdf = norm.cdf(train_acc[index], loc=train_acc_mean, scale=train_acc_var + 1e-6)
        testing_prob = min(testing_cdf, 1 - testing_cdf)
        hypothesis_prob *= testing_prob
        if hypothesis_prob < hypothesis_threshold:
            break
        choice_list.append(train_acc[index])
        train_acc_mean, train_acc_var = np.mean(choice_list), np.var(choice_list) * np.sqrt(len(choice_list))

    choice_list = val_acc[begin_point:-1]
    val_acc_mean, val_acc_var = np.mean(choice_list), np.var(choice_list) * np.sqrt(len(choice_list))
    index = begin_point
    hypothesis_prob = 1
    while -index < len(train_acc):
        index -= 1
        testing_cdf = norm.cdf(train_acc[index], loc=val_acc_mean, scale=val_acc_var + 1e-6)
        testing_prob = min(testing_cdf, 1 - testing_cdf)
        hypothesis_prob *= testing_prob
        if hypothesis_prob < hypothesis_threshold:
            break
        choice_list.append(val_acc[index])
        val_acc_mean, val_acc_var = np.mean(choice_list), np.var(choice_list) * np.sqrt(len(choice_list))
    overfitting_acc_mean = train_acc_mean - val_acc_mean
    overfitting_acc_var = train_acc_var + val_acc_var
    print("overfitting_acc", overfitting_acc_mean, overfitting_acc_var)

    choice_list = []
    for i in range(-begin_point):
        choice_list.append(train_loss[-i - 1])
    # print("train_loss",train_loss[0],train_loss)
    train_loss_mean, train_loss_var = np.mean(choice_list), np.var(choice_list)
    index = begin_point
    hypothesis_prob = 1
    while -index < len(train_acc):
        index -= 1
        testing_cdf = norm.cdf(train_acc[index], loc=train_loss_mean, scale=train_loss_var + 1e-6)
        testing_prob = min(testing_cdf, 1 - testing_cdf)
        hypothesis_prob *= testing_prob
        if hypothesis_prob < hypothesis_threshold:
            break
        choice_list.append(train_loss[index])
        train_loss_mean, train_loss_var = np.mean(choice_list), np.var(choice_list)
    # print(choice_list)

    choice_list = []
    for i in range(-begin_point):
        choice_list.append(val_loss[-i - 1])
    val_loss_mean, val_loss_var = np.mean(choice_list), np.var(choice_list) * np.sqrt(len(choice_list))
    index = begin_point
    hypothesis_prob = 1
    while -index < len(train_acc):
        index -= 1
        testing_cdf = norm.cdf(train_acc[index], loc=val_loss_mean, scale=val_loss_var + 1e-6)
        testing_prob = min(testing_cdf, 1 - testing_cdf)
        hypothesis_prob *= testing_prob
        if hypothesis_prob < hypothesis_threshold:
            break
        choice_list.append(val_loss[index])
        val_loss_mean, val_loss_var = np.mean(choice_list), np.var(choice_list) * np.sqrt(len(choice_list))
    # print(choice_list)
    overfitting_loss_mean = val_loss_mean - train_loss_mean
    overfitting_loss_var = val_loss_var + train_loss_var
    print("overfitting_loss", overfitting_loss_mean, overfitting_loss_var)


if __name__ == '__main__':
    experiments = ["baseline_cutout", "baseline_cutmix", "baseline_mixup", "baseline"]
    alpha_list = [0.1, 1, 10]
    wd_list = [0, 5e-4, 1e-3]

    # for debug
    # train_loss_csv = pd.read_csv("mixup" + '-Loss_train.csv')
    # tran_loss = train_loss_csv['Value'].tolist()
    # print("train_loss", train_loss[-5:-1])
    # val_loss = torch.load('./valid_loss_acc/' + "mixup" + '/val_loss_list', map_location=torch.device('cpu'))
    # print("val_loss",val_loss[-5:-1])

    for experiment in experiments:
        print("experiment name:", experiment)
        if experiment == "baseline":
            for wd in wd_list:
                print("wd", wd)
                train_acc = torch.load('./' + experiment + "/train_acc_list/" + str(wd))
                val_acc = torch.load('./' + experiment + "/val_acc_list/" + str(wd))
                train_loss = torch.load('./' + experiment + "/train_loss_list/" + str(wd))
                val_loss = torch.load('./' + experiment + "/val_loss_list/" + str(wd))
                print("accuracy", max(val_acc))
                get_overfitting_rate(train_acc, val_acc, train_loss, val_loss)
                get_ILV_and_EFR(train_acc, val_acc, train_loss, val_loss)
                draw_acc(train_acc, val_acc, experiment + str(wd))
                draw_loss(train_loss, val_loss, experiment + str(wd))
        else:
            for alpha in alpha_list:
                print("alpha", alpha)
                train_acc = torch.load('./' + experiment + "/train_acc_list/" + str(alpha))
                val_acc = torch.load('./' + experiment + "/val_acc_list/" + str(alpha))
                train_loss = torch.load('./' + experiment + "/train_loss_list/" + str(alpha))
                val_loss = torch.load('./' + experiment + "/val_loss_list/" + str(alpha))
                print("accuracy", max(val_acc))
                get_overfitting_rate(train_acc, val_acc, train_loss, val_loss)
                get_ILV_and_EFR(train_acc, val_acc, train_loss, val_loss)
                draw_acc(train_acc, val_acc, experiment + str(alpha))
                draw_loss(train_loss, val_loss, experiment + str(alpha))

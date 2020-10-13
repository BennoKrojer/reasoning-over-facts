import os
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import config


def plot_tensorflow_log(log_paths, out_dir, iterations_per_epoch, follow_up_run=None, anti=True, no=True,
                        name='Accuracy'):
    for params, (file, iterations_per_epoch) in log_paths.items():
        event_acc = EventAccumulator(str(file))
        event_acc.Reload()

        # Show all tags in the log file
        print(event_acc.Tags()['scalars'])
        try:
            eval_accuracies = event_acc.Scalars('eval_Eval_Accuracy')
        except KeyError:
            eval_accuracies = event_acc.Scalars('eval_Eval_Accuracy_intersected')
        train_accuracies = event_acc.Scalars('eval_Train_Accuracy')
        train_accuracies = [x for x in train_accuracies if x.step / iterations_per_epoch <= 700]
        eval_accuracies = [x for x in eval_accuracies if x.step / iterations_per_epoch <= 700]

        final_metrics = {}

        x = []
        train_acc = []
        val_acc = []

        for event in train_accuracies[:len(train_accuracies)]:
            x.append(event.step / iterations_per_epoch)
            train_acc.append(event.value)
        for event in eval_accuracies[:len(train_accuracies)]:
            val_acc.append(event.value)
        final_metrics['train_acc'] = train_acc[-1]
        final_metrics['eval_acc'] = val_acc[-1]

        plt.plot(x, train_acc, label='SYM train set')
        plt.plot(x, val_acc, label='SYM test set')
        # try:
        #     eval_no_pattern_accuracy = event_acc.Scalars('eval_No_Pattern_Eval_Accuracy')
        #     train_no_pattern_accuracy = event_acc.Scalars('eval_No_Pattern_Train_Accuracy')
        #     eval_no_pat_acc = []
        #     train_no_pat_acc = []
        #     for event in train_no_pattern_accuracy[:len(train_accuracies)]:
        #         train_no_pat_acc.append(event.value)
        #     for event in eval_no_pattern_accuracy[:len(train_accuracies)]:
        #         eval_no_pat_acc.append(event.value)
        #     final_metrics['no_pattern_train_acc'] = train_no_pat_acc[-1]
        #     final_metrics['no_pattern_eval_acc'] = eval_no_pat_acc[-1]
        #     if no:
        #         plt.plot(x, train_no_pat_acc, label='no pattern training accuracy')
        #         plt.plot(x, eval_no_pat_acc, label='no pattern eval accuracy')
        # except KeyError:
        #     print("cant find no pattern")
        #
        try:
            eval_anti_pattern_accuracy = event_acc.Scalars('eval_Anti_Pattern_Eval_Accuracy')
            train_anti_pattern_accuracy = event_acc.Scalars('eval_Anti_Pattern_Train_Accuracy')
            eval_anti_pat_acc = []
            train_anti_pat_acc = []
            for event in train_anti_pattern_accuracy[:len(train_accuracies)]:
                train_anti_pat_acc.append(event.value)
            for event in eval_anti_pattern_accuracy[:len(train_accuracies)]:
                eval_anti_pat_acc.append(event.value)
            final_metrics['anti_pattern_train_acc'] = train_anti_pat_acc[-1]
            final_metrics['anti_pattern_eval_acc'] = eval_anti_pat_acc[-1]
        except KeyError:
            print("no anti")
        if anti:
            plt.plot(x, train_anti_pat_acc, label='ANTISYM train set')
            # plt.plot(x, eval_anti_pat_acc, label='ANTISYM test set')

        plt.xlabel("Epochs")
        plt.ylabel("Precision")
        # plt.locator_params(axis='x', nbins=7)

        plt.legend(loc='bottom right', frameon=True)
        print(final_metrics)

    ax = plt.gca()
    ax.yaxis.label.set_size(25)
    ax.xaxis.label.set_size(20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, name))
    plt.clf()
    # def annot_max(x,y, ax=None):
    #     xmax = x[np.argmax(y[100:])]
    #     ymax = y[100:].max()
    #     text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    #     if not ax:
    #         ax=plt.gca()
    #     bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    #     arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    #     kw = dict(xycoords='data',textcoords="axes fraction",
    #               arrowprops=arrowprops, bbox=bbox_props)
    #     ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
    #
    # plt.plot(x, eval_no_pat_acc, label='No Pattern Accuracy')
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.legend(loc='best', frameon=True)
    # # annot_max(x, np.array(inv_acc))
    # plt.savefig(os.path.join(out_dir, 'InverseAccuracy.png'))
    # plt.clf()
    #
    # plt.plot(x, loss, label='training loss')
    # plt.xlabel("Epochs")
    # plt.ylabel("CrossEntropy-Loss")
    # plt.legend(loc='best', frameon=True)
    # plt.savefig(os.path.join(out_dir, 'Loss.png'))
    # plt.clf()


if __name__ == '__main__':
    log_files = dict()
    log_files['sym anti'] = (Path(
        config.output_dir) / 'runs' / 'symmetry' / 'sampled_20_anti_Jul15_18-51-45',
                                           np.ceil(76800 / 1024))
    #
    # log_files['COMP enhanced test set'] = (Path(
    #     config.output_dir) / 'runs' / 'transitive_enhanced' / 'comp_enh_1temp_2group_20rules_connectedTo_Jul13_12-14-02',
    #                     np.ceil(105200 / 1024))
    # log_files['COMP test set'] = (Path(config.output_dir)/'runs'/'compositional'/'sampled_3layer_Jul09_21-04-35',
    #                              np.ceil(62400/700))
    # log_files['125 attributes'] = (
    # Path(config.output_dir) / 'runs' / 'negation' / '125attri_flayer_Jul16_12-35-21',
    # np.ceil(46400 / 900))
    # log_files['250 attributes'] = (
    # Path(config.output_dir) / 'runs' / 'negation' / '250attri_flayer_Jul16_12-33-10',
    # np.ceil(46400 / 900))
    # log_files['500 attributes'] = (
    #     Path(config.output_dir) / 'runs' / 'negation' / '500attri_flayer_Jul16_12-31-54',
    #     np.ceil(46400 / 900))
    # log_files['1000 attributes'] = (Path(config.output_dir) / 'runs' / 'negation' / 'sampled_4layer_Jul15_18-29-44',
    #                                 np.ceil(46400 / 900))
    out_dir = Path(config.documentation_dir) / 'CoNLL_sampled'
    os.makedirs(str(out_dir), exist_ok=True)
    plot_tensorflow_log(log_files, out_dir, iterations_per_epoch=131, anti=True, no=False,
                        name='SYM_anti1.png')

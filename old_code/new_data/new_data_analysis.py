import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from collections import defaultdict
from scipy.stats import zscore
import copy


class Item:
    def __init__(self, filename, assessment_id, id, score, visible, right_place, drawn_correctly):
        self.filename = filename
        self.assessment_id = assessment_id
        self.id = id
        self.score = float(score)
        self.visible = float(visible)
        self.right_place = float(right_place)
        self.drawn_correctly = float(drawn_correctly)

    def getComputedScore(self):
        if self.visible == 0:
            return 0.0
        else:
            if self.drawn_correctly and self.right_place:
                return 2.0
            elif self.drawn_correctly and not self.right_place:
                return 1.0
            elif not self.drawn_correctly and self.right_place:
                return 1.0
            else:
                return 0.5

    def getScore(self):
        if math.isnan(self.score):
            print("NAN")
        return self.score


class Assessment:
    def __init__(self, id, filename, cheater_score):
        self.items = {}
        self.id = id
        self.filename = filename
        self.person = None
        self.invalid = False
        self.cheater = False
        self.cheater_score = None
        if cheater_score != "NA":
            self.cheater_score = float(cheater_score)
            if self.cheater_score < 0.8:
                self.cheater = True

    def addItem(self, item_id, score, visible, right_place, drawn_correctly):
        if item_id in self.items:
            # check if it's the same (double line) or different (error!)
            curr = self.items[item_id]
            if curr.score == float(score) and curr.visible == float(visible) and curr.right_place == float(right_place) \
                    and curr.drawn_correctly == float(drawn_correctly):
                # nothing to do, just double line
                pass
            else:
                # print("error, item {} already in list of items for figure {}, assessment {}, and is different from current -> remove assessment".format(item_id, self.filename, self.id))
                # mark as invalid -> cannot be used
                self.invalid = True
        else:
            self.items[item_id] = Item(self.filename, self.id, item_id, score, visible, right_place, drawn_correctly)

    def itemCount(self):
        return len(self.items)

    def getScore(self):
        return sum([self.items[i].getScore() for i in self.items])

    def isValid(self):
        # check if all scores and in correct range and not marked as invalid or cheater
        if self.itemCount() == 18 and not self.invalid and not self.cheater:
            return True
        return False

    def setPerson(self, pers):
        self.person = pers

    def getPerson(self):
        return self.person


class Figure:
    def __init__(self, filename, age_group):
        self.filename = filename
        self.assessments = {}
        self.age_group = age_group

    def getAssessment(self, assessment_id, cheater_score):
        if assessment_id not in self.assessments:
            self.assessments[assessment_id] = Assessment(assessment_id, self.filename, cheater_score)
        return self.assessments[assessment_id]

    def getScores(self):
        # only scores of valid assessments
        return [a.getScore() for a in self.getValidAssessments()]

    def getMedianScore(self):
        return np.median(self.getScores())

    def getMedianScoreItems(self):
        # sum of medians scores of items, could be better than bare median score
        assessments = self.getValidAssessments()
        items = defaultdict(list)
        for ass in assessments:
            for i in ass.items:
                items[i].append(ass.items[i].score)
        medians = {}
        for i in items:
            medians[i] = np.median(items[i])
        return sum([medians[m] for m in medians])

    def getMeanScore(self):
        return np.mean(self.getScores())

    def getScoreStd(self):
        var = np.std(self.getScores())
        if var > 5:
            # print("Huge standard deviation in score ({}) for figure {}: {}".format(var, self.filename, self.getScores()))
            pass
        return var

    def getTotalNumberOfAssessments(self):
        # all, also invalid ones
        return len(self.assessments)

    def getNumberOfAssessments(self):
        # only valid assessments
        return len(self.getValidAssessments())

    def getAllAssessments(self):
        return [self.assessments[a] for a in self.assessments]

    def getValidAssessments(self):
        return [self.assessments[a] for a in self.assessments if self.assessments[a].isValid()]

    def hasValidAssessment(self):
        if len(self.getValidAssessments()) > 0:
            return True
        return False

    def __str__(self):
        str = "Figure {}, containing {} assessments " \
            .format(self.filename, len(self.assessments))
        return str


class Person:
    def __init__(self, id, cheater_score):
        self.id = id
        self.assessments = {}
        self.bad = False
        self.cheater_score = None
        if cheater_score != "NA":
            self.cheater_score = float(cheater_score)

    def addAssessment(self, assessment):
        ass_id = assessment.id
        if not ass_id in self.assessments:
            self.assessments[ass_id] = assessment

    def getNumberOfAssessments(self):
        # only valid assessments
        return len(self.getValidAssessments())

    def getValidAssessments(self):
        return [self.assessments[a] for a in self.assessments if self.assessments[a].isValid()]

    def hasValidAssessment(self):
        if len(self.getValidAssessments()) > 0:
            return True
        return False


class Dataset:
    def __init__(self):
        self.figures = {}
        self.persons = {}

    def getAllFigures(self):
        return [self.figures[fig] for fig in self.figures]

    def getAllValidFigures(self):
        return [self.figures[fig] for fig in self.figures if self.figures[fig].hasValidAssessment()]

    def getAllAssessments(self):
        assessments = []
        for fig in self.figures:
            for ass in self.figures[fig].assessments:
                assessments.append(self.figures[fig].assessments[ass])
        return assessments

    def getAllValidAssessments(self):
        assessments = []
        for fig in self.figures:
            for valid in self.figures[fig].getValidAssessments():
                assessments.append(valid)
        return assessments

    def getAllItems(self):
        items = []
        for fig in self.figures:
            for ass in self.figures[fig].assessments:
                for item in self.figures[fig].assessments[ass].items:
                    items.append(self.figures[fig].assessments[ass].items[item])
        return items


def is_number(s):
    try:
        number = float(s)
        return not math.isnan(number)
    except ValueError:
        return False


data = Dataset()

with open('Files_with_adult_children_label.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=';')
    rows = [row for row in reader]
    rows = rows[1:]  # first is label
    age_group = {}
    for ind, row in enumerate(rows):
        filename = row[0]
        age = int(row[1])
        age_group[filename] = age


def find_age_group(filename):
    age = None
    try:
        age = age_group[filename]
    except KeyError:
        if filename.startswith('REF_'):
            alt_filename = filename[4:]
            try:
                age = age_group[alt_filename]
            except KeyError:
                print("2: no age group for {}".format(filename))
        elif filename.startswith('_'):
            alt_filename = filename[1:]
            try:
                age = age_group[alt_filename]
            except KeyError:
                print("3: no age group for {}".format(filename))
        else:
            print("1: no age group for {}".format(filename))
    return age


data_file = 'Data2018-11-14.csv'
data_file = "Data2018-11-29.csv"

with open(data_file) as csv_file:
    labels_reader = csv.reader(csv_file, delimiter=',')
    rows = [row for row in labels_reader]
    rows = rows[1:]  # first is label
    for ind, row in enumerate(rows):
        filename = row[5]
        assessment_id = row[0]
        if not filename in data.figures:
            age = find_age_group(filename)
            data.figures[filename] = Figure(filename, age)

        assessment = data.figures[filename].getAssessment(assessment_id, row[11])
        assessment.addItem(row[6], row[7], row[8], row[9], row[10])

        person_id = row[1]
        if not person_id in data.persons:
            data.persons[person_id] = Person(person_id, row[11])
        data.persons[person_id].addAssessment(assessment)
        assessment.setPerson(data.persons[person_id])

print("{} figures of which {} are valid (1+ valid assessment)".format(len(data.getAllFigures()),
                                                                      len(data.getAllValidFigures())))

print("{} assessments of which {} are valid (18 items)".format(len(data.getAllAssessments()),
                                                               len(data.getAllValidAssessments())))


def assessments_per_figure(figures):
    # number of assessments per figure
    number_of_assessments = [f.getTotalNumberOfAssessments() for f in figures]
    n, bins, patches = plt.hist(x=number_of_assessments, rwidth=0.85,
                                bins=range(min(number_of_assessments), max(number_of_assessments) + 1))
    print("bins: {}".format(bins))
    plt.xlabel('Number of assessments')
    plt.ylabel('Frequency')
    plt.title('Number of assessments per figure'.format(len(assessments), len(figures)))
    maxfreq = n.max()
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


def score_distribution(assessments):
    # score distribution of assessments
    scores = [a.getScore() for a in assessments]
    n, bins, patches = plt.hist(x=scores, rwidth=0.85, bins=36)
    print("bins: {}".format(bins))
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of scores of {} assessments of {} distinct figures'.format(len(assessments), len(figures)))
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


def number_of_figures_against_label(figures):
    # number of figures against median score of that figure
    n_figures_per_label = {}
    for fig in figures:
        score = round(fig.getMedianScore())
        if score in n_figures_per_label:
            n_figures_per_label[score] += 1
        else:
            n_figures_per_label[score] = 1
    n_figures_per_label_counts = [n_figures_per_label[i] for i in n_figures_per_label]
    n_figures_per_label_label = [float(i) for i in n_figures_per_label]

    plt.bar(x=n_figures_per_label_label, height=n_figures_per_label_counts, width=0.85)
    plt.xlabel('Median score of figure')
    plt.ylabel('Number of figures')
    plt.title('Number of figures against label')
    plt.show()


def number_of_assessments_against_label(figures):
    # number of assessments per figure, against median score of that figure
    n_figures_per_label = {}
    counts_per_label = {}
    for fig in figures:
        score = np.round(fig.getMedianScore())
        if score in n_figures_per_label:
            n_figures_per_label[score] += 1
            counts_per_label[score] += fig.getNumberOfAssessments()
        else:
            n_figures_per_label[score] = 1
            counts_per_label[score] = fig.getNumberOfAssessments()
    label_avg_count = [counts_per_label[i] / n_figures_per_label[i] for i in counts_per_label]
    label_label = [float(i) for i in counts_per_label]

    plt.bar(x=label_label, height=label_avg_count)
    plt.xlabel('Median score of figure')
    plt.ylabel('Average number of assessment per figure')
    plt.title('Number of assessments per figure agains label')
    plt.show()


def median_scores(figures):
    median_scores_figures = []
    for fig in figures:
        median_scores_figures.append(fig.getMedianScore())

    n, bins, patches = plt.hist(x=median_scores_figures, rwidth=0.85, bins=36)
    print("bins: {}".format(bins))
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of median scores of {} distinct figures'.format(len(figures)))
    maxfreq = n.max()
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


def std_scores(figures):
    min_number_of_assessments = 5
    figures = [fig for fig in figures if len(fig.getScores()) >= min_number_of_assessments]
    print("Considering {} figures with {}+ assessments".format(len(figures), min_number_of_assessments))
    std_scores_figures = []
    for fig in figures:
        std_scores_figures.append(fig.getScoreStd())

    n, bins, patches = plt.hist(x=std_scores_figures, rwidth=0.85, bins=36)
    print("bins: {}".format(bins))
    plt.xlabel('Empirical standard deviation')
    plt.ylabel('Frequency')
    plt.title('Histogram of empirical std of scores of {} distinct figures ({}+ assessments)'.format(len(figures),
                                                                                                     min_number_of_assessments))
    maxfreq = n.max()
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


def write_data_to_dataloader_format(data):
    # write preprocessing to current dataloader format using median score
    filenames = [fig.filename for fig in data.getAllFigures()]
    label = [fig.getMedianScore() for fig in data.getAllFigures()]
    with open('dlf', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(zip(filenames, label))
    csv_file.close()


def write_human_MSE(data):
    figures = data.getAllFigures()
    figures = [fig for fig in figures if len(fig.getScores()) >= 5]
    print("Considering {} figures with 5+ assessments".format(len(figures)))
    filenames = [fig.filename for fig in figures]
    MSE = []
    for fig in figures:
        errors = []
        score = fig.getMedianScore()
        for ass in fig.getValidAssessments():
            errors.append(ass.getScore() - score)
        MSE.append(np.mean(np.square(np.array(errors))))
    with open('MSEperfigure', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(zip(filenames, MSE))
    csv_file.close()


def write_human_binMSE(data):
    bins = np.array([0, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 100])
    figures = data.getAllFigures()
    figures = [fig for fig in figures if len(fig.getScores()) >= 5]
    print("Considering {} figures with 5+ assessments".format(len(figures)))
    filenames = [fig.filename for fig in figures]
    MSE = []
    for fig in figures:
        errors = []
        scorebin = np.argmin(bins <= fig.getMedianScore()) - 1
        for ass in fig.getValidAssessments():
            assbin = np.argmin(bins <= ass.getScore()) - 1
            errors.append(assbin - scorebin)
        MSE.append(np.mean(np.square(np.array(errors))))
    with open('binMSEperfigure', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(zip(filenames, MSE))
    csv_file.close()


def std_part_scores(items):
    scores_per_part = {}
    for item in items:
        key = (item.filename, item.id)
        if key in scores_per_part:
            scores_per_part[key].append(item.score)
        else:
            scores_per_part[key] = [item.score]

    std_per_part = []
    for key in scores_per_part:
        part = scores_per_part[key]
        std_per_part.append(np.std(part))

    n, bins, patches = plt.hist(x=std_per_part, rwidth=0.85, bins=10)
    print("bins: {}".format(bins))
    plt.xlabel('Empirical standard deviation')
    plt.ylabel('Frequency')
    plt.title(
        'Histogram of STD of item scores of {} item assessments of {} parts'.format(len(items), len(scores_per_part)))
    maxfreq = n.max()
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


def human_loss_mse(data):
    """
    we look at figures with 5+ assessments only
    assumption: median is true label
    """
    figures = data.getAllFigures()
    figures = [fig for fig in figures if len(fig.getScores()) >= 5]
    print("Considering {} figures with 5+ assessments".format(len(figures)))
    errors = []
    errors_vs_score = defaultdict(list)

    for fig in figures:
        score = fig.getMedianScore()
        if score < 5 and False:
            print("true score: {}, assessment scores: {}, mse: {}"
                  .format(score, [a.getScore() for a in fig.getValidAssessments()],
                          np.round(np.mean([np.square(a.getScore() - score) for a in fig.getValidAssessments()]), 2)))

        for ass in fig.getValidAssessments():
            errors.append(ass.getScore() - score)
            errors_vs_score[round(score)].append(np.square(ass.getScore() - score))

    print("Human MSE: {}".format(np.mean(np.square(np.array(errors)))))

    mse_vs_score = -np.ones((37))
    for score in errors_vs_score:
        mse_vs_score[int(score)] = np.mean(errors_vs_score[score])

    # plt.plot(mse_vs_score,'x')
    # plt.xlabel('Median score of figure')
    # plt.ylabel('MSE')
    # plt.title('Human MSE vs score of figure ')
    # plt.show()


def human_loss_binning(data):
    """
    we look at figures with 5+ assessments only
    assumption: median is true label
    """
    figures = data.getAllFigures()
    figures = [fig for fig in figures if len(fig.getScores()) >= 5]
    print("Considering {} figures with 5+ assessments".format(len(figures)))
    errors = []
    errors_vs_score = defaultdict(list)

    bins = np.array([0, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 100])

    for fig in figures:
        score = fig.getMedianScore()
        score_bin = np.argmin(bins <= score) - 1
        for ass in fig.getValidAssessments():
            ass_score_bin = np.argmin(bins <= ass.getScore()) - 1
            errors.append(ass_score_bin - score_bin)

    print("Human binning MSE: {}".format(np.mean(np.square(np.array(errors)))))


def find_outliers(figures):
    high_z_score = []
    fishy_zero_score = []
    for fig in figures:
        for ass in fig.getValidAssessments():
            allOtherScores = fig.getScores()
            allOtherScores.remove(ass.getScore())
            z_score = zscore([ass.getScore()] + allOtherScores)[0]
            if np.abs(z_score) > 3:
                print("Outlier for figure {}. Score vs. other scores: {}/{}, z-score: {}".format(fig.filename,
                                                                                                 ass.getScore(),
                                                                                                 fig.getScores(),
                                                                                                 z_score))
                high_z_score.append(ass)
            if ass.getScore() == 0:
                if len(allOtherScores) > 0 and np.min(allOtherScores) > 8:
                    print("! ", end='')
                    fishy_zero_score.append(ass)
                    # del fig.assessments[ass.id]
                    print("Far-off zero assessment for {}. Score vs. other scores: {}/{}, z-score: {}".format(
                        fig.filename, ass.getScore(), fig.getScores(), z_score))
                # print("Zero assessment (outlier?) for {}. Score vs. other scores: {}/{}, z-score: {}".format(fig.filename, ass.getScore(), fig.getScores(), z_score))
    print(" ")
    print("There are {} assessments with high z-score (outliers?)".format(len(high_z_score)))
    print("There are {} assessments with a fishy zero score far from the other assessments.".format(
        len(fishy_zero_score)))


def assess_quality_of_incomplete_assessments(data):
    # tries to find out if incomplete assessments can be trusted on the level of individual items
    # it compares the item-wise variance of figures considering complete assessments
    # vs. figures with incomplete assessments

    figures = data.getAllFigures()

    # find average variance among individual items for complete assessments
    mean_variances_reference = []
    for fig in figures:
        assessments = fig.getValidAssessments()
        if len(assessments) > 3:
            items = defaultdict(list)
            variance_of_items = {}
            for ass in assessments:
                for i in ass.items:
                    items[i].append(ass.items[i].getScore())
            for i in items:
                variance_of_items[i] = np.var(items[i])
            mean_variances_reference.append(np.mean(list(variance_of_items.values())))
    reference_mean_variance = np.mean(mean_variances_reference)
    print(len(mean_variances_reference))
    print("For figures with 3+ assessments, the average variance among items is {}".format(reference_mean_variance))

    # do the same for incomplete assessments
    mean_variances_incomplete = []
    for fig in figures:
        assessments = [fig.assessments[a] for a in fig.assessments]  # all assessments, including invalid ones
        assessments = [a for a in assessments if not a.cheater]  # exclude cheaters
        for ass in assessments:
            if ass.itemCount() < 18 and len(assessments) > 3:
                items = defaultdict(list)
                variance_of_items = {}
                for ass2 in assessments:
                    for i in ass.items:
                        if i in ass2.items:
                            items[i].append(ass2.items[i].getScore())
                for i in items:
                    variance_of_items[i] = np.var(items[i])
                mean_variances_incomplete.append(np.mean(list(variance_of_items.values())))
    mean_variance_with_incomplete_assessments = np.mean(mean_variances_incomplete)
    print(len(mean_variances_incomplete))
    print(
        "For figures with 3+ assessments (including an incomplete one), the average variance among items is {}".format(
            mean_variance_with_incomplete_assessments))

    # helper function for permutation test
    def perm(a, b):
        combined = np.asarray(a + b)
        sampled_indices = np.random.choice(range(len(combined)), len(combined), replace=False)
        a_new = combined[sampled_indices[0:len(a)]]
        b_new = combined[sampled_indices[len(a):]]
        return list(a_new), list(b_new)

    def stat(a, b):
        return np.mean(a) - np.mean(b)

    def p_value(orig_stat, stats):
        orig_smaller = orig_stat < stats
        p = 1 - (sum(orig_smaller) / len(stats))
        print("p-value: {}".format(p))
        return p

    # permutation test
    # null hypothesis: the mean variances are the same in both cases (difference is zero)
    orig_stat = stat(mean_variances_reference, mean_variances_incomplete)
    stats = []
    for i in range(10000):
        ref, inc = perm(mean_variances_reference, mean_variances_incomplete)
        stats.append(stat(ref, inc))
    n, bins, patches = plt.hist(x=stats, rwidth=0.85, bins=36)
    print("bins: {}".format(bins))
    plt.xlabel('stat')
    plt.ylabel('Frequency')
    plt.title('Permutation test if incomplete assessments have higher variance among items')
    plt.axvline(x=orig_stat)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()
    pval = p_value(orig_stat, stats)
    print("Permutation test if incomplete assessments have higher variance among items: p-value: {}".format(pval))


def assess_quality_of_zero_predictions(data):
    figures = data.getAllFigures()

    all_assessments = []
    zero_score_assessments = []
    for fig in figures:
        if len(fig.getValidAssessments()) >= 5:
            for ass in fig.getValidAssessments():
                difference_to_median = np.abs(ass.getScore() - fig.getMedianScore())
                all_assessments.append(difference_to_median)
                if ass.getScore() == 0:
                    allOtherScores = fig.getScores()
                    allOtherScores.remove(ass.getScore())
                    zero_score_assessments.append(difference_to_median)

    print("There are {} assessments for 5+ figures, of which {} are zero assessments".format(len(all_assessments), len(
        zero_score_assessments)))
    p = 99
    quantile = np.percentile(all_assessments, p)
    print(
        "The {}% quantile of all assessments is at {}. This corresponds to the {}% quantile ({} assessments) of the zero assessments"
        .format(p, quantile, sum(zero_score_assessments > quantile) * 100 // len(zero_score_assessments),
                sum(zero_score_assessments > quantile)))

    plt.hist(x=all_assessments, rwidth=0.85, bins=50, alpha=0.5,
             weights=np.ones(np.asarray(all_assessments).shape) * 1. / len(all_assessments))
    plt.hist(x=zero_score_assessments, rwidth=0.85, bins=50, alpha=0.5,
             weights=np.ones(np.asarray(zero_score_assessments).shape) * 1. / len(zero_score_assessments))
    plt.legend(['All assessments', 'Zero score assessments'], loc='upper right')
    plt.xlabel('Absolute Difference to Median Score')
    plt.ylabel('Fraction')
    plt.title('Histogram of absolute difference to median score of all vs. zero assessments')
    plt.axvline(x=quantile)
    # plt.axvline(x=np.mean(all_assessments))
    plt.show()


def std_of_scores_vs_label(figures):
    min_number_of_assessments = 5
    figures = [fig for fig in figures if len(fig.getScores()) >= min_number_of_assessments]
    print("Considering {} figures with {}+ assessments".format(len(figures), min_number_of_assessments))
    std_scores_figures = defaultdict(list)
    for fig in figures:
        std_scores_figures[fig.getMedianScore()].append(fig.getScoreStd())

    x = []
    y = []
    for label in std_scores_figures:
        x.append(float(label))
        y.append(float(np.mean(std_scores_figures[label])))
    x, y = zip(*sorted(zip(x, y)))

    # helper function to smooth plot
    def smooth(scalars, weight):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value
        return smoothed

    y_smooth = smooth(y, 0.8)

    from scipy.optimize import curve_fit
    def f(x, A, B):  # this is your 'straight line' y=f(x)
        return A * x + B

    A, B = curve_fit(f, x, y)[0]  # your preprocessing x, y to fit

    plt.plot(x, y, ':')  # true values
    plt.plot(x, y_smooth, 'r')  # smoothed line (like tensorboard)
    plt.plot(range(37), f(range(37), A, B))  # least squares fit
    plt.xlabel('Label (complete score)')
    plt.ylabel('Mean Standard Deviation')
    plt.title(
        'Std of Scores vs label ({} figures with {}+ assessments)'.format(len(figures), min_number_of_assessments))
    plt.show()


def difference_median_score(figures):
    # calculates the difference of the median of complete scores vs. the sum of the medians of the individual scores
    # against the label
    min_number_of_assessments = 5
    figures = [fig for fig in figures if len(fig.getScores()) >= min_number_of_assessments]
    print("Considering {} figures with {}+ assessments".format(len(figures), min_number_of_assessments))
    difference = defaultdict(list)
    all_diffs = []
    for fig in figures:
        diff = fig.getMedianScore() - fig.getMedianScoreItems()
        difference[fig.getMedianScore()].append(diff)
        all_diffs.append(diff)

    x = []
    y = []
    for label in difference:
        x.append(float(label))
        y.append(float(np.mean(difference[label])))
    x, y = zip(*sorted(zip(x, y)))

    # helper function to smooth plot
    def smooth(scalars, weight):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value
        return smoothed

    y_smooth = smooth(y, 0.8)

    from scipy.optimize import curve_fit
    def f(x, A, B):  # this is your 'straight line' y=f(x)
        return A * x + B

    A, B = curve_fit(f, x, y)[0]  # your preprocessing x, y to fit

    plt.plot(x, y, ':')  # true values
    plt.plot(x, y_smooth, 'r')  # smoothed line (like tensorboard)
    plt.plot(range(37), f(range(37), A, B))  # least squares fit
    plt.plot(range(-1, 38), np.ones((39)) * np.mean(all_diffs), '-.')  # mean
    plt.xlabel('Label (median score)')
    plt.ylabel('MedianScore - SumOfItemMedians')
    plt.title(
        'Difference Median vs Sum of Item Medians ({} figures with {}+ assessments)'.format(len(figures),
                                                                                            min_number_of_assessments))
    plt.show()

    # overlay of the two histograms
    median_scores = []
    median_scores_items = []
    for fig in figures:
        median_scores.append(fig.getMedianScore())
        median_scores_items.append(fig.getMedianScoreItems())

    bins = np.linspace(-0.1, 36.9, 38)
    plt.hist(x=median_scores, rwidth=0.85, bins=bins, alpha=0.5)
    plt.hist(x=median_scores_items, rwidth=0.85, bins=bins, alpha=0.5)
    plt.legend(['Median Scores', 'Sum of Medians of Item Scores'], loc='upper left')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of median scores of {} distinct figures'.format(len(figures)))
    plt.axvline(x=np.mean(median_scores), color='C0')
    plt.axvline(x=np.mean(median_scores_items), color='C1')
    print("Average difference between median score and sum of item-wise median score: {}".format(
        np.mean(median_scores_items) - np.mean(median_scores)))
    plt.show()


def analyze_persons(data):
    min_number_of_assessments = 5

    all_persons = [data.persons[p] for p in data.persons]
    persons = [data.persons[p] for p in data.persons if data.persons[p].hasValidAssessment()]
    persons_with_many_assessments = [data.persons[p] for p in data.persons if
                                     len(data.persons[p].getValidAssessments()) >= 5]
    cheaters = [data.persons[p] for p in data.persons if
                data.persons[p].cheater_score is not None and data.persons[p].cheater_score < 0.8]
    print(
        "There are {} person of which {} have scored 1+ valid assessment, {} have scored {}+ assessments, and {} are cheaters by client's definition"
        .format(len(all_persons), len(persons), len(persons_with_many_assessments), min_number_of_assessments,
                len(cheaters)))

    # histogram of how many assessments per person
    n_assessments = []
    for pers in all_persons:
        n_assessments.append(pers.getNumberOfAssessments())

    plt.hist(x=n_assessments, rwidth=0.85, bins=30)
    plt.xlabel('Number of assessments')
    plt.ylabel('Frequency')
    plt.title('Histogram of number of assessments of {} persons'.format(len(all_persons)))
    plt.show()

    figures = [fig for fig in data.getAllValidFigures() if len(fig.getScores()) >= min_number_of_assessments]
    # difference to median score
    difference_to_median_score = defaultdict(list)
    for fig in figures:
        for ass in fig.getValidAssessments():
            person = ass.getPerson()
            diff_to_median = np.abs(ass.getScore() - fig.getMedianScore())
            difference_to_median_score[person.id].append(diff_to_median)
    mean_diffs = []
    for p in difference_to_median_score:
        mean_diffs.append(np.mean(difference_to_median_score[p]))
    print("considered {} non-cheaters".format(len(mean_diffs)))

    figures_cheaters = [fig for fig in data.getAllFigures() if
                        len([ass for ass in fig.assessments if
                             not fig.assessments[ass].invalid and fig.assessments[ass].itemCount() == 18])
                        >= min_number_of_assessments]
    print("cheaters have scored {} figures".format(len(figures_cheaters)))
    # difference to median score of cheaters
    difference_to_median_score_cheaters = defaultdict(list)
    for fig in figures_cheaters:
        for ass in fig.getAllAssessments():
            if ass.invalid or ass.itemCount() < 18:
                # only want valid ones
                continue
            if not ass.cheater: continue  # want only cheaters
            person = ass.getPerson()
            diff_to_median = np.abs(ass.getScore() - fig.getMedianScore())
            difference_to_median_score_cheaters[person.id].append(diff_to_median)
    mean_diffs_cheaters = []
    for p in difference_to_median_score_cheaters:
        mean_diffs_cheaters.append(np.mean(difference_to_median_score_cheaters[p]))
    print("considered {} cheaters".format(len(mean_diffs_cheaters)))

    # difference to median score of persons with at least one fishy zero prediction
    fishy_zero_persons = defaultdict(list)
    for fig in figures:
        for ass in fig.getValidAssessments():
            if ass.getScore() == 0:
                allOtherScores = fig.getScores()
                allOtherScores.remove(ass.getScore())
                if len(allOtherScores) > 0 and np.min(allOtherScores) > 3:
                    fishy_zero_persons[ass.getPerson().id].append(ass)
    mean_diffs_fishy_zero = []
    for person in difference_to_median_score:
        if person in fishy_zero_persons:
            mean_diffs_fishy_zero.append(np.mean(difference_to_median_score[person]))
    print("considered {} fishy-zero persons with {} fishy assessments"
          .format(len(mean_diffs_fishy_zero), np.sum([len(fishy_zero_persons[p]) for p in fishy_zero_persons])))

    plt.hist(x=mean_diffs, rwidth=0.85, bins=50, alpha=0.5,
             weights=np.ones(np.asarray(mean_diffs).shape) * 1. / len(mean_diffs))
    plt.hist(x=mean_diffs_cheaters, rwidth=0.85, bins=50, alpha=0.5,
             weights=np.ones(np.asarray(mean_diffs_cheaters).shape) * 1. / len(mean_diffs_cheaters))
    plt.hist(x=mean_diffs_fishy_zero, rwidth=0.85, bins=50, alpha=0.5,
             weights=np.ones(np.asarray(mean_diffs_fishy_zero).shape) * 1. / len(mean_diffs_fishy_zero))
    plt.legend(['All persons', 'Cheaters (< 0.8 avgMatch)', 'Persons with at least one fishy zero prediction'],
               loc='upper left')
    plt.xlabel('Mean difference to median score')
    plt.ylabel('Frequency')
    plt.title('Histogram of mean difference to median score of {} persons'.format(len(persons)))
    quantile = np.percentile(mean_diffs, 95)
    plt.axvline(x=quantile)
    plt.show()


def distribution_of_human_mse(data, system_MSE):
    persons = [data.persons[p] for p in data.persons if data.persons[p].hasValidAssessment() and (
                data.persons[p].cheater_score is None or data.persons[p].cheater_score >= 0.8)]
    min_number_of_assessments = 5
    figures = [fig for fig in data.getAllValidFigures() if len(fig.getScores()) >= min_number_of_assessments]
    difference_to_median_score_honest = defaultdict(list)
    for fig in figures:
        for ass in fig.getValidAssessments():
            person = ass.getPerson()
            diff_to_median = np.square((ass.getScore() - fig.getMedianScore()))
            difference_to_median_score_honest[person.id].append(diff_to_median)
    MSE_person = []
    for person in persons:
        if person.id in difference_to_median_score_honest:
            MSE_person.append(np.mean(difference_to_median_score_honest[person.id]))
    perc = np.sum(np.asarray(MSE_person) > system_MSE) / np.shape(MSE_person)[0]
    print("With the MSE of {} the system performs better than {}% of the people".format(system_MSE, 100 * perc))
    plt.hist(x=MSE_person, bins=50)
    plt.show()


def distribution_of_human_bin_mse(data, system_bin_MSE):
    bins = np.array([0, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 100])
    persons = [data.persons[p] for p in data.persons if data.persons[p].hasValidAssessment() and (
                data.persons[p].cheater_score is None or data.persons[p].cheater_score >= 0.8)]
    min_number_of_assessments = 5
    figures = [fig for fig in data.getAllValidFigures() if len(fig.getScores()) >= min_number_of_assessments]
    difference_to_median_score_honest = defaultdict(list)
    for fig in figures:
        score_bin = np.argmin(bins <= fig.getMedianScore()) - 1
        for ass in fig.getValidAssessments():
            person = ass.getPerson()
            pers_bin = np.argmin(bins <= ass.getScore()) - 1
            diff_to_median = np.square((pers_bin - score_bin))
            difference_to_median_score_honest[person.id].append(diff_to_median)
    MSE_person = []
    for person in persons:
        if person.id in difference_to_median_score_honest:
            MSE_person.append(np.mean(difference_to_median_score_honest[person.id]))
    perc = np.sum(np.asarray(MSE_person) > system_bin_MSE) / np.shape(MSE_person)[0]
    print("With the MSE of {} the system performs better than {}% of the people".format(system_bin_MSE, 100 * perc))
    plt.hist(x=MSE_person, bins=50)
    plt.show()


def distribution_of_human_accurarcy(data, model_accuracy=None):
    bins = np.array([0, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 100])
    persons = [data.persons[p] for p in data.persons if data.persons[p].hasValidAssessment() and (
                data.persons[p].cheater_score is None or data.persons[p].cheater_score >= 0.8)]
    min_number_of_assessments = 5
    figures = [fig for fig in data.getAllValidFigures() if len(fig.getScores()) >= min_number_of_assessments]
    accuracy_honest = defaultdict(list)
    accuracy_all = []
    accuracy_all_fig = []
    agreement = 0
    for fig in figures:
        accuracy_fig = []
        score_bin = np.argmin(bins <= fig.getMedianScore()) - 1
        for ass in fig.getValidAssessments():
            person = ass.getPerson()
            pers_bin = np.argmin(bins <= ass.getScore()) - 1
            correct = 1 * (score_bin == pers_bin)
            accuracy_honest[person.id].append(correct)
            accuracy_all.append(correct)
            accuracy_fig.append(correct)
        if np.mean(accuracy_fig) == 1:
            agreement = agreement + 1
        accuracy_all_fig.append(np.mean(accuracy_fig))
    accuracy_person = []
    for person in persons:
        if person.id in accuracy_honest:
            accuracy_person.append(np.mean(accuracy_honest[person.id]))
    print(np.mean(accuracy_person))
    print(np.mean(accuracy_all))
    print(np.mean(accuracy_all_fig))
    print(agreement / 6363)
    if model_accuracy is not None:
        perc = np.sum(np.asarray(accuracy_person) < model_accuracy) / np.shape(accuracy_person)[0]
        print("With the accuracy of {} the system performs better than {}% of the people".format(model_accuracy,
                                                                                                 100 * perc))
    plt.hist(x=accuracy_person, bins=50)
    plt.show()


def delete_bad_persons(data):
    new_data = copy.deepcopy(data)

    min_number_of_assessments = 5
    figures = [fig for fig in new_data.getAllValidFigures() if len(fig.getScores()) >= min_number_of_assessments]

    # difference to median score
    difference_to_median_score = defaultdict(list)
    for fig in figures:
        for ass in fig.getValidAssessments():
            person = ass.getPerson()
            diff_to_median = np.abs(ass.getScore() - fig.getMedianScore())
            difference_to_median_score[person.id].append(diff_to_median)

    mean_diffs = []
    for p in difference_to_median_score:
        mean_diffs.append(np.mean(difference_to_median_score[p]))

    # get all persons with a high mean difference to median score -> bad quality
    quantile = np.percentile(mean_diffs, 95)
    n_bad_persons = 0
    n_bad_assessments = 0
    for p in difference_to_median_score:
        if np.mean(difference_to_median_score[p]) >= quantile:
            person = new_data.persons[p]
            n_bad_persons += 1
            n_bad_assessments += person.getNumberOfAssessments()
            for ass in person.getValidAssessments():
                ass.invalid = True
            person.bad = True

    print("{} of {} persons are in the quantile, affecting {} assessments".format(n_bad_persons, len(new_data.persons),
                                                                                  n_bad_assessments))

    assessments = data.getAllAssessments()
    figures = data.getAllFigures()
    valid_figures = data.getAllValidFigures()
    valid_assessments = data.getAllValidAssessments()
    persons = data.persons
    good_persons = [p for p in data.persons if not data.persons[p].bad]
    print("Before: {} figure, {} valid figures, {} assessments, {} valid assessments, {} persons, {} good persons"
          .format(len(figures), len(valid_figures), len(assessments), len(valid_assessments), len(persons),
                  len(good_persons)))

    assessments = new_data.getAllAssessments()
    figures = new_data.getAllFigures()
    valid_figures = new_data.getAllValidFigures()
    valid_assessments = new_data.getAllValidAssessments()
    persons = new_data.persons
    good_persons = [p for p in new_data.persons if not new_data.persons[p].bad]
    print("After: {} figure, {} valid figures, {} assessments, {} valid assessments, {} persons, {} good persons"
          .format(len(figures), len(valid_figures), len(assessments), len(valid_assessments), len(persons),
                  len(good_persons)))

    print("\nHuman loss before:")
    human_loss_mse(data)
    human_loss_binning(data)

    print("\nHuman loss after:")
    human_loss_mse(new_data)
    human_loss_binning(new_data)

    return new_data


def score_distribution_adults_vs_children(figures):
    adults = []
    children = []
    unclear = []
    for fig in figures:
        if fig.age_group == 1:
            adults.append(fig)
        elif fig.age_group == 2:
            children.append(fig)
        else:
            unclear.append(fig)
    print("There are {} figures of adults, {} figures of children, and {} unclear figures"
          .format(len(adults), len(children), len(unclear)))

    adults = [fig.getMedianScore() for fig in adults if is_number(fig.getMedianScore())]
    children = [fig.getMedianScore() for fig in children if is_number(fig.getMedianScore())]

    plt.hist(x=adults, rwidth=0.85, bins=np.linspace(0, 36, 37), alpha=0.5, label="Adults",
             weights=np.ones(np.asarray(adults).shape) * 1. / len(adults))
    plt.hist(x=children, rwidth=0.85, bins=np.linspace(0, 36, 37), alpha=0.5, label="Children",
             weights=np.ones(np.asarray(children).shape) * 1. / len(children))
    plt.axvline(x=np.mean(adults), label="Mean Adults", color='C0', alpha=0.5)
    plt.axvline(x=np.mean(children), label="Mean Children", color='C1', alpha=0.5)
    plt.legend(loc='upper left')
    plt.xlabel('Median Score')
    plt.ylabel('Fraction')
    plt.title('Histogram of median scores of adults ({}) vs children ({})'.format(len(adults), len(children)))
    plt.show()


def std_adults_vs_children(figures):
    adults = []
    children = []
    unclear = []
    for fig in figures:
        if fig.age_group == 1:
            adults.append(fig)
        elif fig.age_group == 2:
            children.append(fig)
        else:
            unclear.append(fig)
    print("There are {} figures of adults, {} figures of children, and {} unclear figures"
          .format(len(adults), len(children), len(unclear)))

    adults = [fig.getScoreStd() for fig in adults if is_number(fig.getMedianScore())]
    children = [fig.getScoreStd() for fig in children if is_number(fig.getMedianScore())]

    plt.hist(x=adults, rwidth=0.85, bins=np.linspace(0, 12, 25), alpha=0.5, label="Adults",
             weights=np.ones(np.asarray(adults).shape) * 1. / len(adults))
    plt.hist(x=children, rwidth=0.85, bins=np.linspace(0, 12, 25), alpha=0.5, label="Children",
             weights=np.ones(np.asarray(children).shape) * 1. / len(children))
    plt.axvline(x=np.mean(adults), label="Mean Std Adults", color='C0', alpha=0.5)
    plt.axvline(x=np.mean(children), label="Mean Std Children", color='C1', alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Fraction')
    plt.title(
        'Histogram of standard deviations of scores of adults ({}) vs children ({})'.format(len(adults), len(children)))
    plt.show()


# assessments = preprocessing.getAllAssessments()
# figures = preprocessing.getAllFigures()
# valid_assessments = preprocessing.getAllValidAssessments()

# assessments_per_figure(figures)
# score_distribution(valid_assessments)
# number_of_figures_against_label(figures)
# number_of_assessments_against_label(figures)
# median_scores(figures)
# std_scores(figures)
# find_outliers(figures)
# assess_quality_of_incomplete_assessments(preprocessing)
# assess_quality_of_zero_predictions(preprocessing)
# std_of_scores_vs_label(figures)

# difference_median_score(figures)

# analyze_persons(preprocessing)
# new_data = delete_bad_persons(preprocessing)

# score_distribution_adults_vs_children(figures)
# std_adults_vs_children(figures)

# write_data_to_dataloader_format(preprocessing)

# items = preprocessing.getAllItems()
# std_part_scores(items)

# human_loss_mse(preprocessing)
# human_loss_binning(preprocessing)

# write_human_MSE(preprocessing)

# distribution_of_human_mse(preprocessing, 5.61)
# distribution_of_human_bin_mse(preprocessing, 2.6250550072979975)
# write_human_binMSE(preprocessing)
distribution_of_human_accurarcy(data, 0.3549592)

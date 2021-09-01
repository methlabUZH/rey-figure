from collections import defaultdict
from cv2 import imread
import numpy as np


class ReyFigure:
    """
    Figure class with attributes:
        - filename (*.jpeg)
        - dict of Assessment instances for the figure
    """

    def __init__(self, figure_id, filepath):
        self.figure_id = figure_id
        self.filepath = filepath
        self.assessments = {}

    def get_assessment(self, assessment_id):

        if assessment_id not in self.assessments:
            self.assessments[assessment_id] = Assessment(assessment_id, self.figure_id, self.filepath)

        return self.assessments[assessment_id]

    def get_scores(self):
        # only scores of valid assessments
        return [a.get_score() for a in self.get_valid_assessments()]

    def get_median_score(self):
        return np.median(self.get_scores())

    def get_sum_of_median_item_scores(self):
        return np.sum(self.get_median_score_per_item())

    def get_median_score_per_item(self):
        # get median score of each item
        scores = defaultdict(list)
        list_of_items = [i for i in self.get_valid_assessments()[0].items]
        for assessment in self.get_valid_assessments():
            for i in list_of_items:
                if i in assessment.items:
                    scores[i].append(assessment.items[i].get_score())
        median_scores = [np.median(scores[s]) for s in sorted(scores.keys())]
        return median_scores

    def get_median_part_score_per_item(self):
        # get median score of each part of each item (3 scores per item), as one list of dimension 54
        scores = defaultdict(list)
        list_of_items = [i for i in self.get_valid_assessments()[0].items]
        for assessment in self.get_valid_assessments():
            for i in list_of_items:
                if i in assessment.items:
                    scores[str(i) + "visible"].append(assessment.items[i].visible)
                    scores[str(i) + "right_place"].append(assessment.items[i].right_place)
                    scores[str(i) + "drawn_correctly"].append(assessment.items[i].drawn_correctly)
        median_scores = [np.median(scores[s]) for s in sorted(scores.keys())]
        return median_scores

    def get_image(self):
        image = imread(self.filepath)

        if image is None:
            print("Error: could not load file {}".format(self.filepath))

        return image

    def get_valid_assessments(self):
        return [self.assessments[a] for a in self.assessments if self.assessments[a].is_valid()]

    def has_valid_assessment(self):
        if len(self.get_valid_assessments()) > 0:
            return True
        return False


class Assessment:
    """
    Assessment class with attributes:
        - dict of Item instances for the given figure
        - assessment_id of the assessment
        - filename of the figure
        - whether or not the figure is valid
    """

    def __init__(self, assessment_id, figure_id, filepath):
        self.items = {}
        self.assessment_id = assessment_id
        self.figure_id = figure_id
        self.filepath = filepath
        self.invalid = False

    def add_item(self, item_id, score, visible, right_place, drawn_correct):
        if item_id in self.items:
            # check if it's the same (double line) or different (error!)
            curr = self.items[item_id]

            if (curr.score == float(score)
                    and curr.visible == float(visible)
                    and curr.right_place == float(right_place)
                    and curr.drawn_correct == float(drawn_correct)):
                # nothing to do, just double line
                pass
            else:
                # mark as invalid -> cannot be used
                self.invalid = True
        else:
            self.items[item_id] = Item(figure_id=self.figure_id,
                                       filepath=self.filepath,
                                       assessment_id=self.assessment_id,
                                       item_id=item_id,
                                       score=score,
                                       visible=visible,
                                       right_place=right_place,
                                       drawn_correct=drawn_correct)

    def item_count(self):
        return len(self.items)

    def get_score(self):
        return sum([self.items[i].get_score() for i in self.items])

    def is_valid(self):
        # check if all scores and in correct range and not marked as invalid
        if self.item_count() == 18 and not self.invalid:
            return True
        return False


class Item:
    def __init__(self, figure_id, filepath, assessment_id, item_id, score, visible, right_place, drawn_correct):
        self.figure_id = figure_id
        self.filepath = filepath
        self.assessment_id = assessment_id
        self.item_id = item_id
        self.score = float(score)
        self.visible = float(visible)
        self.right_place = float(right_place)
        self.drawn_correct = float(drawn_correct)

    def get_score(self):
        return self.score

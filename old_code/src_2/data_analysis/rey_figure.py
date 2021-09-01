from collections import defaultdict
import numpy as np
from typing import List

from src.data_analysis.constants import *
from src.data_analysis import Assessment


class ReyFigure:

    def __init__(self, fn: str, age_group: int):
        self.__filename = fn
        self.__age_group = age_group
        self.__assessments = {}

    def get_assessment(self, assessment_id: int, cheater_score: float) -> Assessment:

        if assessment_id not in self.__assessments:
            self.__assessments[assessment_id] = Assessment(assessment_id, self.__filename, cheater_score)

        return self.__assessments[assessment_id]

    def scores(self):
        # only scores of valid assessments
        return [a.compute_score() for a in self.valid_assessments()]

    def median_score(self):
        return np.median(self.scores())

    def median_score_items(self):
        # sum of medians scores of items, could be better than bare median score
        items = defaultdict(list)

        for a in self.valid_assessments():
            for i in a.items:
                items[i].append(a.items[i].score)

        medians = {}
        for i in items:
            medians[i] = np.median(items[i])

        return sum([medians[m] for m in medians])

    def mean_score(self) -> np.ndarray:
        return np.mean(self.scores())

    def score_std(self) -> np.ndarray:
        var = np.std(self.scores())

        if var > LARGE_VAR:
            print(f'large variance (> {LARGE_VAR}) in score ({var}) for figure {self.__filename}')

        return var

    def num_all_assessments(self) -> int:
        # includes invalid assessments
        return len(self.__assessments)

    def num_valid_assessments(self) -> int:
        # only valid assessments
        return len(self.valid_assessments())

    def all_assessments(self) -> List[Assessment]:
        # list of all assessments, including invalid ones
        return [self.__assessments[a] for a in self.__assessments.keys()]

    def valid_assessments(self) -> List[Assessment]:
        return [self.__assessments[a] for a in self.__assessments if self.__assessments[a].is_valid]

    def has_valid_assessment(self) -> bool:

        if len(self.valid_assessments()) > 0:
            return True

        return False

    @property
    def filename(self):
        return self.__filename

    @property
    def age_group(self):
        return self.__age_group

    @property
    def assessments(self):
        return self.__assessments

    def __str__(self):
        return "Figure {}, containing {} assessments ".format(self.__filename, len(self.__assessments))

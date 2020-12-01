from typing import List

from src.data_analysis import ReyFigure
from src.data_analysis import Assessment
from src.data_analysis.item import Item


class Dataset:

    def __init__(self):
        self.__figures = {}
        self.__persons = {}

    def all_figures(self) -> List[ReyFigure]:
        return [self.__figures[fig] for fig in self.__figures]

    def all_valid_figures(self) -> List[ReyFigure]:
        return [self.__figures[fig] for fig in self.__figures if self.__figures[fig].has_valid_assessment()]

    def all_assessments(self) -> List[Assessment]:
        assessment_list = []

        for fig in self.__figures:
            for assessment in self.__figures[fig].all_assessments():
                assessment_list.append(assessment)

        return assessment_list

    def all_valid_assessments(self) -> List[Assessment]:
        assessment_list = []

        for fig in self.__figures:
            for valid_assessment in self.__figures[fig].valid_assessments():
                assessment_list.append(valid_assessment)

        return assessment_list

    def all_items(self) -> List[Item]:
        item_list = []

        for fig in self.__figures:
            for assessment in self.__figures[fig].all_assessments():
                for _, itm in self.__figures[fig].assessments[assessment].items.items():
                    item_list.append(itm)

        return item_list

    def num_figures(self):
        return len(self.all_figures())

    def num_valid_figures(self):
        return len(self.all_valid_figures())

    def num_assessments(self):
        return len(self.all_assessments())

    def num_valid_assessments(self):
        return len(self.all_valid_assessments())

    @property
    def figures(self):
        return self.__figures

    @property
    def persons(self):
        return self.__persons

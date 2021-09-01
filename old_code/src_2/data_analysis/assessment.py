from src.data_analysis.item import Item
from src.data_analysis.constants import *


class Assessment:

    def __init__(self, assessment_id: int, fn: str, cheater_score):
        self.__items = {}
        self.__id = assessment_id
        self.__filename = fn
        self.__person = None
        self.__inconsistent_rows = False
        self.__is_valid = True
        self.__is_cheater = False
        self.__cheater_score = None

        if cheater_score != "NA":
            self.__cheater_score = float(cheater_score)

            if self.__cheater_score < 0.8:
                self.__is_cheater = True

    def add_item(self, item_id, score, visible, right_place, drawn_correctly):
        if item_id in self.__items:
            # check if it's the same (double line) or different (error!)
            itm = self.__items[item_id]

            if (itm.score == float(score) and itm.visible == float(visible) and itm.right_place == float(right_place)
                    and itm.drawn_correctly == float(drawn_correctly)):
                # nothing to do, just double line
                pass

            else:
                # mark as invalid -> cannot be used
                self.__inconsistent_rows = True
        else:
            self.__items[item_id] = Item(self.__filename, self.__id, item_id, score, visible, right_place,
                                         drawn_correctly)

    def num_items(self):
        return len(self.__items)

    def compute_score(self):
        return sum([self.__items[i].getScore() for i in self.__items.keys()])

    def check_if_valid(self):
        # check if all scores and in correct range and not marked as invalid or cheater
        if self.num_items() == NUM_ITEMS and not self.__inconsistent_rows and not self.__is_cheater:
            self.__is_valid = True
        else:
            self.__is_valid = False

        return self.__is_valid

    @property
    def id(self):
        return self.__id

    @property
    def is_valid(self):
        self.check_if_valid()
        return self.__is_valid

    @property
    def items(self):
        return self.__items

    @property
    def person(self):
        return self.__person

    @person.setter
    def person(self, pers):
        self.__person = pers

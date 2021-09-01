import math


class Item:

    def __init__(self, fn: str, assessment_id: int, item_id: float, score: float, visible: float, right_place: float,
                 drawn_correct: int):
        self.__filename = fn
        self.__assessment_id = assessment_id
        self.__item_id = item_id
        self.__score = float(score)
        self.__visible = float(visible)
        self.__right_place = float(right_place)
        self.__drawn_correctly = float(drawn_correct)

    def compute_score(self):
        if self.__visible == 0:
            return 0.0
        else:
            if self.__drawn_correctly and self.__right_place:
                return 2.0
            elif self.__drawn_correctly and not self.__right_place:
                return 1.0
            elif not self.__drawn_correctly and self.__right_place:
                return 1.0
            else:
                return 0.5

    @property
    def score(self):
        if math.isnan(self.__score):
            print('NAN')
        return self.__score

    @property
    def visible(self):
        return self.__visible

    @property
    def right_place(self):
        return self.__right_place

    @property
    def drawn_correctly(self):
        return self.__drawn_correctly

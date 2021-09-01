from src.data_analysis import Assessment


class Person:
    def __init__(self, person_id, cheater_score):
        self.__id = person_id
        self.__assessments = {}
        self.__bad = False
        self.__cheater_score = None

        if cheater_score != "NA":
            self.cheater_score = float(cheater_score)

    def add_assessment(self, assessment: Assessment):

        if assessment.id in self.__assessments:
            return

        self.__assessments[assessment.id] = assessment

    def num_assessments(self):
        return

    def valid_assessments(self):
        return [self.__assessments[a] for a in self.__assessments if self.__assessments[a].is_valid]

    def has_valid_assessment(self):
        if len(self.valid_assessments()) > 0:
            return True

        return False

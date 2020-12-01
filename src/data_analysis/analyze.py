import csv

from src.data_analysis import ReyFigure
from src.data_analysis.dataset import Dataset
from src.data_analysis import Person

data_file = '/Users/maurice/Desktop/Data07112018.csv'
age_group_file = 'Files_with_adult_children_label.csv'


def find_age_group(filename, age_group: dict):
    age = None
    try:
        age = age_group[filename]

    except KeyError:
        if filename.startswith('REF_'):
            alt_filename = filename[4:]
            try:
                age = age_group[alt_filename]
            except KeyError:
                # print("2: no age group for {}".format(filename))
                pass
        elif filename.startswith('_'):
            alt_filename = filename[1:]
            try:
                age = age_group[alt_filename]
            except KeyError:
                # print("3: no age group for {}".format(filename))
                pass
        else:
            # print("1: no age group for {}".format(filename))
            pass

    return age


def main():
    data = Dataset()

    age_group = {}
    try:
        with open(age_group_file) as csv_file:
            reader = csv.reader(csv_file, delimiter=';')
            rows = [row for row in reader]
            rows = rows[1:]  # first is label
            for ind, row in enumerate(rows):
                filename = row[0]
                age = int(row[1])
                age_group[filename] = age

    except FileNotFoundError:
        print('file with age groups not found!')

    with open(data_file) as csv_file:

        labels_reader = csv.reader(csv_file, delimiter=',')
        rows = [row for row in labels_reader]
        rows = rows[1:]  # first is header

        for ind, row in enumerate(rows):

            # parse row
            assessment_id = row[0]
            person_id = row[1]
            filename = str(row[5])
            item_id = int(row[6])
            score = float(row[7])
            visible = int(row[8])
            right_place = int(row[9])
            drawn_correctly = int(row[10])
            cheater_score = row[11]

            if filename not in data.figures:
                age = find_age_group(filename, age_group)
                data.figures[filename] = ReyFigure(filename, age)

            assessment = data.figures[filename].get_assessment(assessment_id, cheater_score)
            assessment.add_item(item_id, score, visible, right_place, drawn_correctly)

            if person_id not in data.persons:
                data.persons[person_id] = Person(person_id, cheater_score)

            data.persons[person_id].add_assessment(assessment)
            assessment.person = data.persons[person_id]

    print(f'{data.num_figures()} figures of which {data.num_valid_figures()} are valid (1+ valid assessment)')
    print(f'{data.num_assessments()} assessments of which {data.num_valid_assessments()} are valid (18 items)')


if __name__ == '__main__':
    main()

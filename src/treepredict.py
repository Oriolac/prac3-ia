import sys
from decisionnode import DecisionNode

def read(file_name):
    file = open(file_name, 'r')
    part = []
    for line in file:
        example = line.split('\t')
        example[3]= int(example[3])
        example[4] = example[4].strip()
        part.append(example)
    return part, len(part)

    # TODO
    pass


def unique_counts(part):
    dict = {}
    for entry in part:
        if entry[4] not in dict:
            dict[entry[4]] = 1
        else:
            dict[entry[4]] = dict.get(entry[4]) + 1
    return dict


def gini_impurity(data_set):
    num_entries = len(data_set)
    results = unique_counts(data_set)
    imp = 0
    key_list = results.keys()
    sum = 0
    for key in key_list:
        sum += (results.get(key)/num_entries)*(results.get(key)/num_entries)
    imp = 1 - sum

    return float(imp)


def entropy(data_set):
    from math import log
    num_entries = len(data_set)
    results = unique_counts(data_set)
    key_list = results.keys()
    sum = 0
    for key in key_list:
        prob = results.get(key)/num_entries
        log2 = log(prob, 2)
        sum += prob*log2

    return -sum


# Partition the data set in 2 partitions depending on column and value
def divideset(part, column, value):
    set1 = []
    set2 = []
    for entry in part:
        if isinstance(value, int) or isinstance(value, float):
            if entry[column] <= value:
                set1.append(entry)
            else:
                set2.append(entry)
        else:
            if entry[column] == value:
                set1.append(entry)
            else:
                set2.append(entry)
    return set1, set2


def decreaseofimpurity(total_imp, prop_l, left_imp, prop_r, right_imp):
    return total_imp - prop_l*left_imp - prop_r*right_imp


def get_columns(part):
    diff_columns = []
    num_columns = len(part[0]) - 1
    for column in range(num_columns):
        for row in range(len(part)):
            if (column, part[row][column]) not in diff_columns:
                diff_columns.append((column, part[row][column]))
    return diff_columns

def buildtree(part, scoref=entropy, beta=0):
    if len(part) == 0:
        return DecisionNode()
    current_score = scoref(part)

    columns_to_analize = get_columns(part)

    # Set up some variables to track the best criteria
    best_gain = 0
    best_criteria = None
    best_sets = None

    for elem in columns_to_analize:
        t_set, f_set = divideset(part, elem[0], elem[1])
        p_true = len(t_set)/len(part)
        p_false = len(f_set)/len(part)
        current_gain = decreaseofimpurity(current_score, p_true, scoref(t_set), p_false, scoref(f_set))

        if current_gain > best_gain:
            best_gain = current_gain
            best_criteria = elem
            best_sets = (t_set, f_set)

    if best_gain >= beta and best_criteria is not None:
        return DecisionNode(best_criteria[0], best_criteria[1], None, buildtree(best_sets[0], scoref, beta), buildtree(best_sets[1], scoref, beta))
    else:
        return DecisionNode(results=unique_counts(part))


def printtree(tree, indent=''):
    # Is this a leaf node?
    if tree.results is not None:
        print(indent+str(tree.results))
    else:
        # Print the criteria
        print(indent + str(tree.col)+':'+str(tree.value)+'? ')
        # Print the branches
        print(indent+'T->')
        printtree(tree.tb, indent+'  ')
        print(indent+'F->')
        printtree(tree.fb, indent+'  ')


if __name__ == '__main__':
    # Read input file and save in [[]] and num of entries
    data_set, num_entries = read(sys.argv[1])
    # Get a dictionary with key: class_name, value: total
    class_dict = unique_counts(data_set)
    # Get Gini impurity
    gini_impurity = gini_impurity(data_set)
    # Get entropy
    entropy = entropy(data_set)
    tree = buildtree(data_set)
    printtree(tree)







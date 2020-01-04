import sys
from decisionnode import DecisionNode

def read(file_name):
    file = open(file_name, 'r')
    part = []
    for line in file:
        example = line.split('\t')
        example[3] = int(example[3])
        example[4] = example[4].strip()
        part.append(example)
    return part, len(part)

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

def get_column_values(part, column):
    values = []
    for row in part:
        if not row[column] in values:
            values.append(row[column])
    return values

def get_best_gain(part, scoref):
    best_gain = 0
    best_criteria = None
    best_sets = None
    current_score = scoref(part)
    for column in range(0, len(part[0])- 1):
        values = get_column_values(part, column)
        for value in values:
            set1, set2 = divideset(part, column, value)
            gain = current_score - len(set1)/len(part) * scoref(set1) - len(set2)/len(part) * scoref(set2)
            if best_gain < gain:
                best_gain = gain
                best_criteria = (column, value)
                best_sets = (set1, set2)
    return best_gain, best_criteria, best_sets
        
def it_buildtree(part, scoref=entropy, beta=0):
    stack=[]
    stackDef = []
    stack.append(part)
    while len(stack) != 0:
        conjunt = stack.pop(-1)
        best_gain, best_criteria, best_sets = get_best_gain(conjunt, scoref)
        if best_sets != None and best_sets[0] != None:
            stack.append(best_sets[0])
        if best_sets != None and best_sets[1] != None:
            stack.append(best_sets[1])
        stackDef.append((best_gain, best_criteria, best_sets, conjunt))

    accumulativeNodes = []
    while len(stackDef) != 0:
        best_gain, best_criteria, best_sets, conjunt = stackDef.pop(-1)
        if best_gain > beta:
            tree2 = accumulativeNodes.pop(-1)
            tree1 = accumulativeNodes.pop(-1)
            accumulativeNodes.append(DecisionNode(col=best_criteria[0], value=best_criteria[1], tb=tree1, fb=tree2))
        else:
            accumulativeNodes.append(DecisionNode(results=unique_counts(conjunt)))
    return accumulativeNodes.pop()

def get_best_gain2(part, scoref):
    best_gain = 0
    best_criteria = None
    best_sets = None

    current_score = scoref(part)
    columns_to_analize = get_columns(part)

    for elem in columns_to_analize:
        t_set, f_set = divideset(part, elem[0], elem[1])
        p_true = len(t_set)/len(part)
        p_false = len(f_set)/len(part)

        current_gain = decreaseofimpurity(current_score, p_true, scoref(t_set), p_false, scoref(f_set))

        if current_gain > best_gain:
            best_gain = current_gain
            best_criteria = elem
            best_sets = (t_set, f_set)
    
    return best_gain, best_criteria, best_sets

def it_buildtree2(part, scoref=entropy, beta=0):
    stack = []
    stack.append((part, None))
    while len(stack) != 0:
        part, parent = stack.pop()
        if len(part) != 0:

            best_gain, best_criteria, best_sets = get_best_gain2(part, scoref)
            
            if best_gain >= beta and best_criteria is not None:
                current_node = DecisionNode(col=best_criteria[0], value=best_criteria[1])
                stack.append((best_sets[0], current_node))
                stack.append((best_sets[0], current_node))
            else:
                current_node = DecisionNode(results=unique_counts(part))
            if parent != None:
                if parent.tb == None:
                    parent.tb = current_node
                else:
                    parent.fb = current_node
            if parent == None:
                superparent = current_node
    return superparent



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
    gini = gini_impurity(data_set)
    # Get entropy
    entrop = entropy(data_set)
    tree = it_buildtree(data_set, scoref=gini_impurity)
    printtree(tree)







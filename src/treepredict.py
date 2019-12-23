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


def read_car_data(file_name):
    file = open(file_name, 'r')
    part = []
    for line in file:
        obj = line.split(',')
        obj[6] = obj[6].strip()
        part.append(obj)
    return part, len(part)


def unique_counts(part):
    dict = {}
    for entry in part:
        last_col = len(entry)-1
        if entry[last_col] not in dict:
            dict[entry[last_col]] = 1
        else:
            dict[entry[last_col]] = dict.get(entry[last_col]) + 1
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
    """Retorna una llista de tuples amb totes les categories i els seus valors, sense
    repeticions. Ex: [(0, 'slashdot'), ...]"""
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


def classify(obj, tree):
    return tree.get_leaf_node(obj)


def test_performance(testset, testset_len, trainingset):
    tree = buildtree(trainingset)
    num_correct = 0
    for object in testset:
        real_result = object[len(object)-1]
        result_dict = classify(object, tree)
        obtained_result = list(result_dict.keys())[0]
        if real_result == obtained_result:
            num_correct += 1

    return num_correct/testset_len


def test_111():
    # Read input file and save in [[]] and num of entries
    data_set, num_entries = read(sys.argv[1])
    # Get a dictionary with key: class_name, value: total
    class_dict = unique_counts(data_set)
    # Get Gini impurity
    gini = gini_impurity(data_set)
    # Get entropy
    entr = entropy(data_set)
    tree = buildtree(data_set, scoref=gini_impurity)
    #printtree(tree)
    return tree


def test_113(tree):
    new_object = ['google', 'UK', 'yes', 25]
    # new_object = ['google', 'UK', 'no', 17]
    print("Result partition: " + str(classify(new_object, tree)))


def test_114():
    # Training set with 4 examples of each class
    train_data_set1, train_num_entries1 = read_car_data("data_sets/trainingset-car1.data")
    # Training set with 8 examples of each class
    train_data_set2, train_num_entries2 = read_car_data("data_sets/trainingset-car2.data")
    # Training set with 12 examples of each class
    train_data_set3, train_num_entries3 = read_car_data("data_sets/trainingset-car3.data")
    # Training set with 16 examples of each class
    train_data_set4, train_num_entries4 = read_car_data("data_sets/trainingset-car4.data")
    # Training set with 20 examples of each class
    train_data_set5, train_num_entries5 = read_car_data("data_sets/trainingset-car5.data")
    # Training set with 166 entries
    train_data_set6, train_num_entries6 = read_car_data("data_sets/trainingset-car6.data")
    test_data_set, test_num_entries = read_car_data("data_sets/testset-car.data")
    print("Accuracy 1: " + str(test_performance(test_data_set, test_num_entries, train_data_set1)) + ", entries: " + str(train_num_entries1))
    print("Accuracy 2: " + str(test_performance(test_data_set, test_num_entries, train_data_set2)) + ", entries: " + str(train_num_entries2))
    print("Accuracy 3: " + str(test_performance(test_data_set, test_num_entries, train_data_set3)) + ", entries: " + str(train_num_entries3))
    print("Accuracy 4: " + str(test_performance(test_data_set, test_num_entries, train_data_set4)) + ", entries: " + str(train_num_entries4))
    print("Accuracy 5: " + str(test_performance(test_data_set, test_num_entries, train_data_set5)) + ", entries: " + str(train_num_entries5))
    print("Accuracy 6: " + str(test_performance(test_data_set, test_num_entries, train_data_set6)) + ", entries: " + str(train_num_entries6))


def num_prototypes(dict):
    """Returns the total number of prototypes in a dictionary"""
    sum = 0
    for entry in dict:
        sum += dict[entry]

    return sum


def buildtree_prune(part, scoref=entropy, beta=0):
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

    if best_gain >= beta and best_criteria is not None and num_prototypes(unique_counts(part)) >= 5:
        return DecisionNode(best_criteria[0], best_criteria[1], None, buildtree_prune(best_sets[0], scoref, beta), buildtree_prune(best_sets[1], scoref, beta))
    else:
        return DecisionNode(results=unique_counts(part))


def mergedicts (dict1, dict2):
    dict = {}

    for key in dict1:
        dict[key] = dict1[key]
    for key in dict2:
        if key in dict:
            dict[key] += dict2[key]
        else:
            dict[key] = dict2[key]
    return dict


def entropy_dict(dict):
    from math import log
    num_entries = num_prototypes(dict)
    key_list = dict.keys()
    sum = 0
    for key in key_list:
        prob = dict.get(key)/num_entries
        log2 = log(prob, 2)
        sum += prob*log2
    return -sum


def prunetree(tree, threshold):
    if tree.get_child(True).is_leaf_node() and tree.get_child(False).is_leaf_node():
        #print("Parent with leaf childs")
        tbranch_entropy = entropy(tree.get_child(True).results)
        fbranch_entropy = entropy(tree.get_child(False).results)
        merged_dicts = mergedicts(tree.get_child(True).results, tree.get_child(False).results)
        union_entropy = entropy_dict(merged_dicts)
        sum_entropies = tbranch_entropy+fbranch_entropy
        if union_entropy > sum_entropies + threshold:
            print(tbranch_entropy)
            print(fbranch_entropy)
            print(union_entropy)
            print("threshold: ", float(sum_entropies+threshold))
            print("Pruned !")
            print("col: ", tree.col)
            print("val: ", tree.value)
            return DecisionNode(results=merged_dicts)
        else:
            #print("Not pruned!")
            return tree
    elif tree.results is None:
        #print("Internal node")
        return DecisionNode(tree.col, tree.value, None, prunetree(tree.get_child(True), threshold),
                            prunetree(tree.get_child(False), threshold))

    else:
        #print("Leaf node")
        return tree


def test_116():
    train_data_set, train_num_entries = read_car_data("data_sets/trainingset-car3.data")
    # Build completely the decision tree
    tree = buildtree_prune(train_data_set)
    printtree(tree)
    pruned_tree = prunetree(tree, 0.4)
    printtree(pruned_tree)


if __name__ == '__main__':
    # *** 1.1.1 ***
    # tree = test_111()
    # *** 1.1.3 ***
    #test_113(tree)
    # *** 1.1.4 ***
    #test_114()
    # *** 1.1.6 ***
    test_116()










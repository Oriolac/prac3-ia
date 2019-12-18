class DecisionNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb

    def get_column(self):
        return self.col

    def get_value(self):
        return self.value

    def get_result(self):
        return self.results

    def get_true_branch(self):
        return self.tb

    def get_false_branch(self):
        return self.fb

    def is_true(self, obj):

        obj_value = obj[self.col]

        if isinstance(self.value, int) or isinstance(self.value, float):
            return obj_value <= self.value
        else:
            return obj_value == self.value

    def get_child(self, branch):
        if branch is True:
            col = self.tb.get_column()
            val = self.tb.get_value()
            result = self.tb.get_result()
            tb = self.tb.get_true_branch()
            fb = self.tb.get_false_branch()
            return DecisionNode(col, val, result, tb, fb)
        else:
            col = self.fb.get_column()
            val = self.fb.get_value()
            result = self.fb.get_result()
            tb = self.fb.get_true_branch()
            fb = self.fb.get_false_branch()
            return DecisionNode(col, val, result, tb, fb)

    def get_leaf_node(self, obj):
        if self.get_result() is not None:
            return self.get_result()
        else:
            if self.is_true(obj):
                return self.get_child(True).get_leaf_node(obj)
            else:
                return self.get_child(False).get_leaf_node(obj)

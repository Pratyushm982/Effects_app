class Param:
    pass


class NumericParam(Param):
    def __init__(self, name, value, min_val, max_val, step=0.01):
        self.name = name
        self.value = value
        self.min = min_val
        self.max = max_val
        self.step = step
        self.type = "numeric"


class ChoiceParam(Param):
    def __init__(self, name, value, choices):
        self.name = name
        self.value = value
        self.choices = choices
        self.type = "choice"

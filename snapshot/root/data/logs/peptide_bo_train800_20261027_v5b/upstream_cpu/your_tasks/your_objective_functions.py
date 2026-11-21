import numpy as np
from apex_oracle import apex_wrapper
from apex_oracle.refseqs import REFERENCE_SEQUENCE

class ObjectiveFunction:
    """Objective function f, we seek x that MAXIMIZE f(x)"""

    def __init__(self):
        pass

    def __call__(self, x_list):
        """Input
            x_list:
                a LIST of input space items from the origianl input
                search space (i.e. list of aa seqs)
        Output
            scores_list:
                a LIST of float values obtained by evaluating your
                objective function f on each x in x_list
                or np.nan in the wherever x is an invalid input
        """
        return self.query_black_box(x_list)

    def query_black_box(self, x_list):
        """Input
            x_list: a list of input space items x_list
        Output
            scores_list:
                a LIST of float values obtained by evaluating your
                objective function f on each x in x_list
                or np.nan in the wherever x is an invalid input
        """
        raise NotImplementedError('Must implement method query_black_box() for the black box objective')

class ExampleObjective(ObjectiveFunction):
    """Example objective funciton length of the input space items
    This is just a dummy example where the objective is the
    avg number of A and M's in the sequence
    """

    def __init__(self):
        super().__init__()

    def query_black_box(self, x_list):
        scores_list = []
        for x in x_list:
            if type(x) != str:
                score = np.nan
            elif len(x) == 0:
                score = 0
            else:
                score = 0
                for char in x:
                    if char in ['A', 'M']:
                        score += 1
                score = score / len(x)
            scores_list.append(score)
        return scores_list

class ApexObjective(ObjectiveFunction):
    """Objective function for optimizing against the provided APEX Oracle"""

    def __init__(self, score_version='mean'):
        self.score_version = score_version
        self.cache = {}
        self.refseqs = REFERENCE_SEQUENCE
        super().__init__()

    def query_black_box(self, x_list):
        if len(x_list) == 0:
            return []
        x_list = [x[:150] for x in x_list]
        unseen_sequences = [x for x in x_list if x not in self.cache]
        if unseen_sequences:
            new_scores = apex_wrapper(unseen_sequences)
            for x, score in zip(unseen_sequences, new_scores):
                self.cache[x] = score
        scores = [self.cache[x] for x in x_list]
        scores = np.array(scores)
        if self.score_version == 'bacteria_0':
            scores = -scores[:, 0]
        else:
            assert 0
        return scores.tolist()
'Objective functions with unique string identifiers \nidentifiers can be passed in when running LOL-BO or ROBOT with --task_id arg\nwhcih specifies which diversity function to use \n--task_specific_args can be used to specify a list of args passed into the init of \nany of these objectives when they are initialized \n'
OBJECTIVE_FUNCTIONS_DICT = {'example': ExampleObjective, 'apex': ApexObjective}

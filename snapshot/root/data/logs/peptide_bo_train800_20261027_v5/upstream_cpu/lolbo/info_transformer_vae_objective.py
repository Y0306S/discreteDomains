import random
import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(file_dir)
sys.path.append(f'{parent_dir}')
import numpy as np
import torch
from lolbo.latent_space_objective import LatentSpaceObjective
from uniref_vae.load_vae import load_vae
from your_tasks.your_blackbox_constraints import CONSTRAINT_FUNCTIONS_DICT
from your_tasks.your_objective_functions import OBJECTIVE_FUNCTIONS_DICT
ALL_AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
PATH_TO_VAE_STATE_DICT = '../uniref_vae/saved_models/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118.pkl'

class ApexConstrainedDiverseObjective(LatentSpaceObjective):
    """Objective class supports all optimization tasks using the
    InfoTransformerVAE"""

    def __init__(self, similarity: float | None=None, template_id=None, task_id='apex', task_specific_args=None, divf_id='edit_dist', path_to_vae_statedict=PATH_TO_VAE_STATE_DICT, xs_to_scores_dict=None, max_string_length=50, num_calls=0, constraint_function_ids=None, constraint_thresholds=None, constraint_types=None):
        if constraint_types is None:
            constraint_types = []
        if constraint_thresholds is None:
            constraint_thresholds = []
        if constraint_function_ids is None:
            constraint_function_ids = []
        if xs_to_scores_dict is None:
            xs_to_scores_dict = {}
        if task_specific_args is None:
            task_specific_args = []
        self.path_to_vae_statedict = path_to_vae_statedict
        self.task_specific_args = task_specific_args
        self.max_string_length = max_string_length
        self.divf_id = divf_id
        assert task_id in OBJECTIVE_FUNCTIONS_DICT
        self.objective_function = OBJECTIVE_FUNCTIONS_DICT[task_id](*self.task_specific_args)
        self.template_id = template_id
        self.similarity = similarity
        self.constraint_functions = []
        for ix, constraint_threshold in enumerate(constraint_thresholds):
            cfunc_class = CONSTRAINT_FUNCTIONS_DICT[constraint_function_ids[ix]]
            cfunc = cfunc_class(threshold_value=constraint_threshold, threshold_type=constraint_types[ix])
            self.constraint_functions.append(cfunc)
        super().__init__(num_calls=num_calls, xs_to_scores_dict=xs_to_scores_dict, task_id=task_id)

    def ensure_one_C(self, seq):
        """Ensure only one 'C' is in the sequence."""
        while seq.count('C') > 1:
            index = random.choice([i for i, letter in enumerate(seq) if letter == 'C'])
            replacement = random.choice([aa for aa in ALL_AMINO_ACIDS if aa != 'C'])
            seq = seq[:index] + replacement + seq[index + 1:]
        return seq

    def vae_decode(self, z):
        """Input
            z: a tensor latent space points
        Output
            a corresponding list of the decoded input space
            items output by vae decoder
        """
        if type(z) is np.ndarray:
            z = torch.from_numpy(z).float()
        z = z.cpu()
        self.vae = self.vae.eval()
        self.vae = self.vae.cpu()
        sample = self.vae.sample(z=z)
        decoded_seqs = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]
        temp = []
        for seq in decoded_seqs:
            seq = seq.replace('X', 'A')
            seq = seq.replace('-', '')
            seq = self.ensure_one_C(seq)
            if len(seq) == 0:
                seq = 'AAA'
            temp.append(seq)
        decoded_seqs = temp
        return decoded_seqs

    def query_oracle(self, x):
        """Input:
            a single input space item x
        Output:
            method queries the oracle and returns
            the corresponding score y,
            or np.nan in the case that x is an invalid input
        """
        return self.objective_function(x)

    def initialize_vae(self):
        """Sets self.vae to the desired pretrained vae and
        sets self.dataobj to the corresponding data class
        used to tokenize inputs, etc."""
        self.vae, self.dataobj = load_vae(path_to_vae_statedict=self.path_to_vae_statedict, max_string_length=self.max_string_length)
        print('max string length: ', self.vae.max_string_length)
        sys.stdout.flush()

    def vae_forward(self, xs_batch):
        """Input:
            a list xs
        Output:
            z: tensor of resultant latent space codes
                obtained by passing the xs through the encoder
            vae_loss: the total loss of a full forward pass
                of the batch of xs through the vae
                (ie reconstruction error)
        """
        tokenized_seqs = self.dataobj.tokenize_sequence(xs_batch)
        encoded_seqs = [self.dataobj.encode(seq).unsqueeze(0) for seq in tokenized_seqs]
        X = self.dataobj.collate_fn(encoded_seqs)
        dict = self.vae(X.cpu())
        vae_loss, z = (dict['loss'], dict['z'])
        return (z, vae_loss)

    @torch.no_grad()
    def compute_constraints(self, xs_batch):
        """Input:
            a list xs (list of sequences)
        Output:
            c: tensor of size (len(xs),n_constraints) of
                resultant constraint values, or
                None of problem is unconstrained
                Note: constraints, must be of form c(x) <= 0!
        """
        if len(self.constraint_functions) == 0:
            return None
        all_cvals = []
        for cfunc in self.constraint_functions:
            cvals = cfunc(xs_batch)
            all_cvals.append(cvals)
        return torch.cat(all_cvals, -1)

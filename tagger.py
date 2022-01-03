from typing import Dict, List, Tuple
from enum import Enum

class SpecialState(Enum):
    BEGIN = 1
    END = 2


class TransitionProbs:

    def __init__(self, transition_probs: Dict[str, Dict[str, float]]):
        self.transition_probs = transition_probs

    def get_prob(self, tag: str, antecedent: str):
        
        tag_subdict = self.transition_probs.get(tag)

        if tag_subdict is None:
            raise ValueError(f"Tag {tag} not found in transition probs")

        transition_prob = tag_subdict.get(antecedent)

        if transition_prob is None:
            return 0
        return transition_prob


class ObservationProbs:

    def __init__(self, observation_probs: Dict[str, Dict[str, float]]):
        self.observation_probs = observation_probs

    def get_prob(self, observation: str, tag: str):
        
        observation_subdict = self.observation_probs.get(observation)

        # TODO: Handle OOV cases
        if observation_subdict is None:
            raise ValueError(f"Observation {observation} not found in observation probs")

        observation_prob = observation_subdict.get(tag)

        if observation_prob is None:
            return 0

        return observation_prob

transition_probs = TransitionProbs({
    "VB": {
        "<s>": .019,
        "VB": .0038,
        "TO": .83,
        "NN": .004,
        "PPSS": .23
    },
    "TO": {
        "<s>": .043,
        "VB": .035,
        "NN": .016,
        "PPSS": .00079 
    },
    "NN": {
        "<s>": .041,
        "VB": .047,
        "TO": .00047,
        "NN": .087,
        "PPSS": .0012
    },
    "PPSS": {
        "<s>": .067,
        "VB": .0070,
        "NN": .0045,
        "PPSS": 0.00014 
    },
})

observation_probs = ObservationProbs({
    "I": {
        "PPSS": .37
    },
    "want": {
        "VB": .0093,
        "NN": .000054
    },
    "to": {
        "TO": .99
    },
    "race": {
        "VB": .00012,
        "NN": .00057
    }
})


states = ["VB", "TO", "NN", "PPSS"]


class ViterbiState:
    """ Class that maintains the current state of a viterbi forward pass.
    Comes with a function that allows backtracking the pointers after the forward pass
    is finished
    """

    def __init__(self):
        self.viterbi_matrix = []
        self.backpointer_matrix = []
        self.processed_observations = []

    def backtrack(self, states: List[str], final_pointer: int):
        # backtracking
        prediction = []
        pointer = final_pointer
        for i in range(len(self.processed_observations)-1, -1, -1):
            prediction.append(states[pointer])
            pointer = self.backpointer_matrix[i][pointer]

        prediction.reverse()

        return prediction


class ViterbiProcessor:
    """ Class that holds the main functionalies for POS taggin with an HMM
    """

    def __init__(self, states: List[str], observation_probs: ObservationProbs, transition_probs: TransitionProbs):
        self.states = states
        self.observation_probs = observation_probs
        self.transition_probs = transition_probs

    def initialization(self, viterbi_state: ViterbiState, observation: str):
        """ Perform the initialization step

        Args:
            viterbi_state (ViterbiState): An ViterbiState instance in its inital state
            observation (str): The first observation (token) in the sequence to be tagged
        """
        assert len(viterbi_state.processed_observations) == 0, "State is not in initial state"

        viterbi_row = []
        backpointer_row = []

        for state in states:
            transition_prob = self.transition_probs.get_prob(state, "<s>")
            observation_prob = self.observation_probs.get_prob(observation, state)

            viterbi = transition_prob * observation_prob

            viterbi_row.append(viterbi)
            backpointer_row.append(SpecialState.BEGIN)

        # update viterbi state
        viterbi_state.viterbi_matrix.append(viterbi_row)
        viterbi_state.backpointer_matrix.append(backpointer_row)
        viterbi_state.processed_observations.append(observation)

    def recursion(self, viterbi_state: ViterbiState, observation: str,
                  timestep: int) -> Tuple[List[float], List[int]]:

        assert timestep > 0, "Timestep in recursion step cannot be less than 1"

        viterbi_row = []
        backpointer_row = []

        for current_state in self.states:
            max_viterbi = 0
            backpointer = None
            # calculate viterbi scores and assign the backpointers
            for (i, prev_state) in enumerate(self.states):
                transition_prob = self.transition_probs.get_prob(current_state, prev_state)
                observation_prob = self.observation_probs.get_prob(observation, current_state)
                viterbi = viterbi_state.viterbi_matrix[timestep-1][i] * transition_prob * observation_prob
                if viterbi > max_viterbi:
                    max_viterbi = viterbi
                    backpointer = i

            viterbi_row.append(max_viterbi)
            backpointer_row.append(backpointer)

        # update viterbi state
        viterbi_state.viterbi_matrix.append(viterbi_row)
        viterbi_state.backpointer_matrix.append(backpointer_row)
        viterbi_state.processed_observations.append(observation)

    def termination(self, viterbi_state: ViterbiState, timestep: int) -> int:
        """ Termination step. So far, we don't treat the special final state
        and assume that all tokens have equal probability to finish a sequence.

        Args:
            viterbi_state (ViterbiState): The current viterbi state
            timestep (int): The final timestep

        Returns:
            backpointer (int): The final backpointer
        """
        assert timestep == len(viterbi_state.processed_observations), "Timestep does not match number of processed observations"

        max_viterbi = 0
        for i in range(len(self.states)):
            viterbi = viterbi_state.viterbi_matrix[timestep-1][i]
            if viterbi > max_viterbi:
                max_viterbi = viterbi
                backpointer = i

        return backpointer
                
    def predict(self, observations: List[str]):
        """ Orchestrating method for the viterbi process

        Args:
            observations (List[str]): A list of observations
        """

        viterbi_state = ViterbiState()
        
        for i, o in enumerate(observations):
            # initialization
            if i == 0:
                self.initialization(viterbi_state, o)
                continue

            # recursion
            self.recursion(viterbi_state, o, i)

        final_pointer = self.termination(viterbi_state, i+1)

        pred = viterbi_state.backtrack(self.states, final_pointer)

        return pred


if __name__ == "__main__":
    viterbi = ViterbiProcessor(states, observation_probs, transition_probs)

    observations = ["I", "want", "to", "race"]
    print(observations)
    prediction = viterbi.predict(observations)
    print(prediction)

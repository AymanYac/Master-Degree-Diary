import sys

class Lazy_DFA:
    current_state = None
    previous_states = []
    def __init__(self, states, alphabet, transition_function, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.start_state = start_state
        self.accept_states = accept_states
        self.current_state = start_state
        return
    
    def transition_to_state_with_input(self, input_value):
        if ((self.current_state, input_value) not in self.transition_function.keys()):
            ##self.current_state = None
            self.current_state=start_state
            return
        self.current_state = self.transition_function[(self.current_state, input_value)]
        return
    
    def in_accept_state(self):
        return self.current_state in accept_states
    
    def go_to_initial_state(self):
        self.current_state = self.start_state
        return

def process(bit,elem,count):
	if(line.split(" ")[0] == '0'):


if __name__ == "__main__":
	query=sys.argv[2]
	path=sys.argv[1]


	with open(path,'r') as f:
		count=0
		for line in f:
			process(line.split(" ")[0],line.split(" ")[1].strip("\n"),count)
			if(line.split(" ")[0] == '0'):
				count+=1
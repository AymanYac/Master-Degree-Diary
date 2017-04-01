import sys

class Lazy_DFA:
	current_state = None
	previous_states = []
	trans_table = dict()
	def __init__(self,e1):
		current_state=1
		
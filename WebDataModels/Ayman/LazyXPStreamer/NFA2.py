import sys
class Lazy_DFA:
    current_state=None
    previous_states=[]
    trans_table={}
    succ = False
    def __init__(self,e1):
        self.current_state=1
        self.trans_table['1']={}
        self.trans_table['1'][e1]='2'
    def process(self,ei,qi,flag):
        if(flag==0):
            if ei in self.trans_table[str(self.current_state)].keys():
                self.previous_states.append(self.current_state)
                self.current_state=self.trans_table.get(str(self.current_state),{}).get(ei)
                self.succ=False
                if(self.current_state == '2'):
                    self.trans_table[str(self.current_state)]={}
                    self.trans_table['2'][ei]=2
                    self.succ=False
                if(qi!="#"):
                    self.trans_table[str(self.current_state)]={}
                    self.trans_table[str(self.current_state)][qi]=int(self.current_state)+1
                    self.succ=False
                else:
                    self.trans_table[str(self.current_state)]={}
                    self.succ=True
            else:
                self.previous_states.append(self.current_state)
                current_state=1
        else:
            current_state=self.previous_states.pop()

if __name__ == "__main__":
    query=sys.argv[2]
    path=sys.argv[1]
    stem=query.strip("//")
    split=stem.split("/")
    dfa = Lazy_DFA(split[0])
    print(dfa.trans_table)
    for i in range(0,len(split)-1):
        dfa.process(split[i],split[i+1],0)
        print(dfa.trans_table)
    dfa.process(split[i+1],'#',0)
    print(dfa.trans_table)
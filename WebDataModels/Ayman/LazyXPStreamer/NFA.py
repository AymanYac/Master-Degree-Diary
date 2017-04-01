import sys

class transition_builder:
    current_state=None
    previous_states=[]
    trans_table={}

    def __init__(self,e1):
        self.current_state=1
        self.trans_table['1']={}
        self.trans_table['1'][e1]='2'
    def advance_table(self,ei,ej):
            if ei in self.trans_table[str(self.current_state)].keys():
                self.previous_states.append(self.current_state)
                self.current_state=self.trans_table.get(str(self.current_state),{}).get(ei)
                self.trans_table[str(self.current_state)]={}

                if(self.current_state == '2'):

                    self.trans_table['2'][ei]=2
                if(ej!="#"):
                    self.trans_table[str(self.current_state)][ej]=int(self.current_state)+1
                else:
                    pass
            else:
                self.previous_states.append(self.current_state)
                current_state=1

def create_table(split,limit):

    e1=split[0].strip("//")

    tb = transition_builder(e1)
    #print(dfa.trans_table)
    if(limit-1 ==len(split)):
        for i in range(0,len(split)-1):
            tb.advance_table(split[i],split[i+1])
            #print(dfa.trans_table)

        tb.advance_table(split[i+1],'#')
        return tb.trans_table
    else:
        for i in range(0,limit-1):
            tb.advance_table(split[i],split[i+1])
        return tb.trans_table




if __name__ == "__main__":
    query=sys.argv[2]
    path=sys.argv[1]
    stem=query.strip("//")
    split=stem.split("/")
    print(create_table(split,5))
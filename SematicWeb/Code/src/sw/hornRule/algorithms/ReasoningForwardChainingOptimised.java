/**
 * 
 */
package sw.hornRule.algorithms;

import sw.hornRule.models.FactBase;
import sw.hornRule.models.Formalism;

/**
 * @author  <Your name>
 *
 */
public class ReasoningForwardChainingOptimised extends AlogrithmChaining {
	
	/**
	 * @param a knowledge base ruleBase (in a given formalism)
	 * @param a base of facts : factBase (in a given formalism)
	 * @return the saturation of KB w.r.t. facts (the minimal fix point of KB from facts)
	 */
	public FactBase forwardChainingOptimise(Formalism ruleBase, Formalism factBase){
		//It's your turn to implement the algorithm
		return null;
	};

 
	public boolean entailment(Formalism ruleBase, Formalism factBase, Formalism query) {
		FactBase allInferredFacts = forwardChainingOptimise(ruleBase, factBase);
		if(allInferredFacts.getFact().contains(query))
			return true;
		else
			return false;
	}

	@Override
	public int countNbMatches() {
		// TODO Auto-generated method stub
		return 0;
	}

}

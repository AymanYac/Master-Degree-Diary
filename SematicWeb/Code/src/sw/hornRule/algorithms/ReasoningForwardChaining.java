/**
 * 
 */
package sw.hornRule.algorithms;

import sw.hornRule.models.FactBase;
import sw.hornRule.models.HornRule;
import sw.hornRule.models.HornRuleBase;
import sw.hornRule.models.Variable;
import sw.hornRule.models.Formalism;
import java.util.Iterator;
/**
 * @author  <YACHAOUI AYMAN>
 *
 */
public class ReasoningForwardChaining extends AlogrithmChaining {
 
	/**
	 * @param a knowledge base kb (in a given formalism)
	 * @param facts (in a given formalism)
	 * @return forwardChaining(ruleBase,factBase), also called the saturation of ruleBase w.r.t. factBase, 
	 * mathematically it computes the minimal fix point of KB from facts)
	 */
	//It's your turn to implement the algorithm, including the methods match() and eval()
	public FactBase forwardChaining(Formalism ruleBase, Formalism factBase){
		FactBase baseFaits = (FactBase) factBase;
		HornRuleBase baseRegles =(HornRuleBase)ruleBase;
		FactBase FB = new FactBase();
		int i = 1;
		int preItSize,postItSize;
		do{
			Iterator<HornRule> baseReglesIter = baseRegles.getRules().iterator();
			System.out.println("iteration "+i+": ");
			FB.setFact(baseFaits.getFact());
			preItSize = FB.getFact().size();
			while(baseReglesIter.hasNext()) {
				HornRule R = baseReglesIter.next();
				if(eval(R,FB)){
					for(Variable clause: R.getConclusions()){
						baseFaits.getFact().add(clause);
					}
					baseReglesIter.remove();
					
				}
			}
			i = i+1;
			postItSize = baseFaits.getFact().size();
		}while(preItSize!=postItSize);
		return new FactBase(baseFaits.getFact());
	};
	public boolean match(Variable I, FactBase FB){
		nbMatches+=1;
		for(Variable clause: FB.getFact())
			if(clause.toString().equals(I.toString())) return true;
		return false;
		
	}
	public boolean eval(HornRule R, FactBase FB){
		for(Variable fait:R.getConditions()){
			if(!match(fait,FB)) return false;
		}
		return true;	
	}

	public boolean entailment(Formalism ruleBase, Formalism factBase, Formalism query) 	{
		FactBase allInferredFacts = forwardChaining(ruleBase, factBase);
		for(Variable fait : allInferredFacts.getFact()){
			if(fait.toString().equals(query.toString()))
			return true;
		}
		return false;
	}

	@Override
	//It's your turn to implement this method
	public int countNbMatches() {
		return nbMatches;
	}

}

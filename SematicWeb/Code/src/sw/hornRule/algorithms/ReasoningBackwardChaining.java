/**
 * 
 */
package sw.hornRule.algorithms;

import sw.hornRule.models.Variable;

import java.util.HashSet;

import sw.hornRule.models.FactBase;
import sw.hornRule.models.Formalism;
import sw.hornRule.models.HornRule;
import sw.hornRule.models.HornRuleBase;

/**
 * @author  <YACHAOUI Ayman>
 *
 */
public class ReasoningBackwardChaining extends AlogrithmChaining {

	public boolean entailment(Formalism ruleBase, Formalism factBase, Formalism query) {
		return backwardChaining(ruleBase,factBase,query);
	}

	public boolean backwardChaining(Formalism ruleBase, Formalism factBase,
			Formalism query) {
		FactBase baseFaits = (FactBase)factBase;
		HornRuleBase baseRegles = (HornRuleBase) ruleBase;
		FactBase baseFaitIter = baseFaits;
		Formalism queryIter = (Formalism) query;
		
		if(match(queryIter,baseFaitIter)) return true;
		else {
			for(HornRule regle: baseRegles.getRules()){
				if(match(queryIter, regle.getConclusions())){
					boolean def = true;
					int i = 1;
					while(def && i<=regle.getConditions().size()){
						for(Variable cond : regle.getConditions()){
							def = backwardChaining(baseRegles,baseFaitIter,cond);
						}
						i++;
					}
					if(def) return true;
				}
			}
			return false;
		}
	}
	private boolean match(Formalism queryIter, FactBase baseFaitIter) {
		nbMatches++;
		boolean b = baseFaitIter.getFact().toString().contains(queryIter.toString());
		return b;
	}
	private boolean match(Formalism queryIter, HashSet<Variable> ConcRegle) {
		boolean b = ConcRegle.toString().contains(queryIter.toString());
		return b;
	}

	@Override
	public int countNbMatches() {
		return nbMatches;
	}
	}

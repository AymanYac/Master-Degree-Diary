package sw.hornRule.algorithms;

import sw.hornRule.models.Formalism;

/**
 * The contracts that define the functions of a reasoner
 * 
 * @author Yue Ma
 *
 */
public interface Reasoner {

	/**
	 * return true iff (ruleBase U factBase) |= query
	 */
	public abstract boolean entailment(Formalism ruleBase, Formalism factBase, Formalism query);

	/**
	 * return true iff ruleBase U factBase is consistent, iff (ruleBase U factBase) |= False 
	 */
	public abstract boolean consistent(Formalism ruleBase, Formalism factBase); 
	
}

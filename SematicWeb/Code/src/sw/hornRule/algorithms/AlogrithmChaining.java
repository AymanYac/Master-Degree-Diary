package sw.hornRule.algorithms;

import sw.hornRule.models.Formalism;
import sw.hornRule.models.Variable;

/**
 * @author Yue Ma
 *
 */
public abstract class AlogrithmChaining implements Reasoner{
	
	protected int nbMatches; // le nombre de match(l,BF)
	
	public AlogrithmChaining() {
		this.nbMatches = 0;
	}

	public int getNbMatches() {
		return nbMatches;
	}
	
	public abstract int countNbMatches();
	
	public abstract boolean entailment(Formalism ruleBase, Formalism factBase, Formalism query);
	
	/**
	 * return true iff ruleBase U factBase is consistent, iff (ruleBase U factBase) |= False
	 */
	public boolean consistent(Formalism ruleBase, Formalism factBase){
		if(entailment(ruleBase, factBase, Variable.False))
			return true;	
		else 
			return false;
	}
}

/**
 * 
 */
package sw.hornRule.algorithms;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Scanner;

import sw.hornRule.models.FactBase;
import sw.hornRule.models.Formalism;
import sw.hornRule.models.HornRule;
import sw.hornRule.models.HornRuleBase;
import sw.hornRule.models.Variable;

/**
 * @author  <YACHAOUI Ayman>
 *
 */
public class ReasoningBackwardChainingwithQuestions extends AlogrithmChaining {

	public boolean user(Formalism queryIter, HornRuleBase baseReglesIter) {
		Iterator<HornRule> regle = baseReglesIter.getRules().iterator();
		HashSet<Variable> clauseDroites;
		while(regle.hasNext()){
			HornRule reg = regle.next();
			clauseDroites = reg.getConclusions();
			boolean drapeau = false;
			for(Variable conc : clauseDroites){
				drapeau = conc.toString().equals(queryIter.toString());
				if(drapeau) return false;
			}
		}
		return true;
	}
	public boolean interaction(Formalism queryIter){
		boolean drapeau = true;
		Scanner sc = new Scanner(System.in);
		System.out.println(queryIter.toString());
		System.out.println("True or False ? ");
		String input = sc.next();
		if(input.equals("True")){
			drapeau = true;
		}else if (input.equals("False")){
			drapeau = false;
		}
		sc.close();
		return drapeau;
	}
	@Override
	public boolean entailment(Formalism ruleBase, Formalism factBase, Formalism Query) {
		// TODO To complete
		// When a literal (i.e. a variable or its negation) cannot be replied by deductive reasoning, 
		// it will be asked to users to give an answer (if the liter holds according to the user)
		FactBase baseFaits = (FactBase)factBase;
		HornRuleBase baseRegles = (HornRuleBase) ruleBase;
		FactBase baseFaitsIter = baseFaits;
		Formalism queryIter = (Formalism) Query;
		if(match(queryIter,baseFaitsIter)) return true;
		else {
			ArrayList<HornRule> baseReglesConcluables = match(queryIter,baseRegles);
			for(HornRule regle: baseReglesConcluables){
				HashSet<Variable> condset = regle.getConditions();
				Iterator<Variable> condsetIter = condset.iterator();
				boolean drapeau = true;
				while(drapeau && condsetIter.hasNext()){
					Variable cond = condsetIter.next();
					drapeau = entailment(baseRegles,baseFaitsIter,cond);
				}
				if(drapeau) return true;
			}
			if(user(queryIter,baseRegles)){
				return interaction(queryIter);

			}
			else return false;
		}
	}
 
	private ArrayList<HornRule> match(Formalism queryIter, HornRuleBase baseReglesIter) {
		ArrayList<HornRule> reglesConcluables = new ArrayList<HornRule>();
		Iterator<HornRule> regleIter = baseReglesIter.getRules().iterator();
		HashSet<Variable> clausesDroites;
		boolean drapeau=false;
		while(regleIter.hasNext()){
			HornRule regle = regleIter.next();
			clausesDroites = regle.getConclusions();
			for(Variable cd : clausesDroites){
				drapeau = cd.toString().equals(queryIter.toString());
				if(drapeau) reglesConcluables.add(regle);
			}
		}
		return reglesConcluables;
	}
	
	private boolean match(Formalism queryIter, FactBase regleFaitsIter) {
		nbMatches++;
		for(Variable fait: regleFaitsIter.getFact())
			if(fait.toString().equals(queryIter.toString())) return true;
		return false;
	}

	@Override
	public int countNbMatches() {
		return nbMatches;
	}

}

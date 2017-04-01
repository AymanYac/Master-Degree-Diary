package problem.hornRules;
 
import java.io.OutputStream;
import java.util.HashSet;

import sw.hornRule.algorithms.*;
import sw.hornRule.models.*;

public class ReasoningHorn {

	public static void main(String[] args) {
		
		ReasoningForwardChaining reasoner = new ReasoningForwardChaining();
		Tutorial1 pb = new Tutorial1();
		HornRuleBase kb = pb.getRuleBase();
		FactBase fb = pb.getFactBase();
		
		for(HornRule r: kb.getRules()){
			System.out.println(r);
		}
		System.out.print("\nThe fact base is: \n");
		System.out.print(fb);
		
		//Display all facts inferred by the given knowledge base kb and fact base fb
		HashSet<Variable> inferredAllFacts = reasoner.forwardChaining(kb,fb).getFact();
		System.out.println("All the inferred facts are:");
		for(Variable s: inferredAllFacts){
			System.out.print(s);
		}
		
		Variable q = new Variable("transoceanic_race");
		if(reasoner.entailment(kb, fb, q))
			System.out.println("\nYes, the query is entailed by the given rules and facts");
		else
			System.out.println("\nNo, the query is not entailed based on the given rules and facts");
	}
}

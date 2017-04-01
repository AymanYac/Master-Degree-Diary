package problem.hornRules;

import java.util.ArrayList;
import java.util.HashSet;

import sw.hornRule.models.*;

/**
 * @author Yue Ma
 *
 */
public class Tutorial1 extends Variable{
	
	HashSet<String> variableNames; //to record all variables appearing in the following rules
	String[] rulestring = {"if boat and sport and sail then sailboat", 
			"if boat and pleasure and sail then sailboat",
			"if sailboat and not_triangular_latin_sail then gaff_rig",
			"if keel then keelboat",
			"if not_keel and sailboat then sailing_dinghy and portable",
			"if habitable and sailboat then sailboat_cruise and not_portable",
			"if not_longer_than_8 then not_longer_than_13 and not_longer_than_10",
			"if keelboat then not_portable",
			"if longer_than_13 then longer_than_10",
			"if keelboat and not_keelboat_regatte then keelboat_cruise",
			"if keelboat and sport then keelboat_sport",
			"if longer_than_10 then longer_than_8",
			"if keelboat and not_habitable then keelboat_regatte",
			"if not_portable and not_habitable and sailboat then sailboat_promenade",
			"if keelboat_cruise then sailboat_cruise",
			"if sailboat_cruise and longer_than_8 and not_longer_than_10 then cruise_semi_offshore",
			"if sailboat_cruise and longer_than_10 then racing_can and cruise_offshore",
			"if sailboat_cruise and not_longer_than_8 then sailboat_cruise_coastal",
			"if sailboat_cruise and number_of_shells_larger_than_1 then sailboat_multishell",
			"if longer_than_13 and racing_can then transoceanic_race"};
	String[] factstring = {"longer_than_13", "habitable", "not_keel", "boat", "sport", "sail"};

	FactBase baseFact;
	HornRuleBase ruleBase;

	Tutorial1() {
		variableNames = new  HashSet<String>();
		
		// create the base of Horn rules
		ruleBase = new HornRuleBase();
		ArrayList<HornRule> rules = ruleBase.getRules();
		for(String r: rulestring){
			HornRule eachrule = new HornRule();
			HashSet<Variable> condition = new HashSet<Variable>();
			HashSet<Variable> conclusion = new HashSet<Variable>();
			String[] conditionsarray = r.substring(3, r.indexOf("then")-1).split("and");
			for(String s:conditionsarray){
				condition.add(new Variable(s.trim()));		
				variableNames.add(s.trim());
			}
			String[] conclusionsarray = r.substring(r.indexOf("then")+5).split("and");
			for(String s:conclusionsarray){
				conclusion.add(new Variable(s.trim()));
				variableNames.add(s.trim());
			}			
			eachrule.setConditions(condition);
			eachrule.setConclusions(conclusion);
			rules.add(eachrule);
		}
		
		//create the base of facts
				baseFact = new FactBase();
				HashSet<Variable> facts = baseFact.getFact();
				
				for(String s: factstring){
					facts.add(new Variable(s.trim()));
				} 
		
	}

	public FactBase getFactBase() {
		return baseFact;
	}

	public void setFactBase(FactBase baseFact) {
		this.baseFact = baseFact;
	}

	public HornRuleBase getRuleBase() {
		return ruleBase;
	}

	public void setRuleBase(HornRuleBase ruleBase) {
		this.ruleBase = ruleBase;
	}	
	
	
}

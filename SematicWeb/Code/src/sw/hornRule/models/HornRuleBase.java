package sw.hornRule.models;
 
import java.util.ArrayList;
import java.util.HashSet;

public class HornRuleBase extends HornRule{
	
	ArrayList<HornRule> rules;
	
	public HornRuleBase() {
		super(); 
		this.rules = new ArrayList<HornRule>();
	}

	public HornRuleBase(HashSet<Variable> conditions,
			HashSet<Variable> conclusions, ArrayList<HornRule> rules) {
		super(conditions, conclusions);
		this.rules = rules;
	}

	public ArrayList<HornRule> getRules() {
		return rules;
	}

	public void setRules(ArrayList<HornRule> rules) {
		this.rules = rules;
	}

	@Override
	public String toString() {
		return "RuleBaseHorn [rules=" + rules + "]";
	}
	
	
	
}

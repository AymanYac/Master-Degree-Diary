package sw.hornRule.models;

import java.util.HashSet;

/**
 * @author 
 *
 */
public class HornRule extends Formalism{
	HashSet<Variable> conditions;
	HashSet<Variable> conclusions;
	
	public HornRule(HashSet<Variable> conditions,
			HashSet<Variable> conclusions) {
		super();
		this.conditions = conditions;
		this.conclusions = conclusions;
		this.name = "Horn";
	}

	public HornRule() {
		super();
		this.conditions = null;
		this.conclusions = null;
		this.name = "Horn";
	}

	public HashSet<Variable> getConditions() {
		return conditions;
	}

	public void setConditions(HashSet<Variable> conditions) {
		this.conditions = conditions;
	}

	public HashSet<Variable> getConclusions() {
		return conclusions;
	}

	public void setConclusions(HashSet<Variable> conclusions) {
		this.conclusions = conclusions;
	}

	@Override
	public String toString() {
		return "HornRule: if " + conditions + ", then "
				+ conclusions + ".";
	}
	public boolean inConditions(Variable v){
		for(Variable clause : this.conditions){
			if(clause.toString().equals(v.toString())) return true;
		}
		return false;
	}

	public boolean inConclusions(Variable v){
		for(Variable clause : this.conclusions){
			if(!clause.toString().equals(v.toString())) return true;
		}
		return false;
	}
	
	
}

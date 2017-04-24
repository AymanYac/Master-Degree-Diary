package constants;

public class Formating {

	
	public static final String transformStringForURL(String input){
		return input.trim().replaceAll("\\s+","+");
	}
	
	public static final String getFileNameForInputs(String...inputs){
		StringBuffer buff=new StringBuffer();
		boolean first=true;
		for(String input: inputs){
			if(!first) buff.append("_");
			else first=false;
			buff.append(transformStringForURL(input));
		}
		buff.append(".xml");
		return buff.toString();
	}
	
}

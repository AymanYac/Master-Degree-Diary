package download;

public class Triple {
   /** by convention an element (be it subject, object or predicate)  
    *  is a variable iff it starts with the character ?
    * **/ 
	public final String subject;
	public final String predicate;
	public final String object;
	
	public Triple(String subject, String predicate, String object){
		this.subject=subject;
		this.predicate=predicate;
		this.object=object;
	}
}

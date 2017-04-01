import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import parsers.ParseResultsForWS;
import parsers.WebServiceDescription;
import download.WebService;


public class Main {

	public static final void main(String[] args) throws Exception{
		
		List<String> params = Arrays.asList("http://musicbrainz.org/ws/1/artist/?name=", null);
		

	    //Testing without loading the description of the WS
	    /** WebService ws=new WebService("mb_getArtistInfoByName",params);	
	    String fileWithCallResult = ws.getCallResult("Frank Sinatra");
		System.out.println("The call is   **"+fileWithCallResult+"**");
		ws.getTransformationResult(fileWithCallResult);**/
		
		
		//Testing with loading the description WS
	    WebService ws=WebServiceDescription.loadDescription("mb_getArtistInfoByName");
	    
	    String fileWithCallResult = ws.getCallResult("Leonard Cohen");
		System.out.println("The call is   **"+fileWithCallResult+"**");
		String fileWithTransfResults=ws.getTransformationResult(fileWithCallResult);
		ArrayList<String[]>  listOfTupleResult= ParseResultsForWS.showResults(fileWithTransfResults, ws);
		
		
		System.out.println("The tuple results are ");
		for(String [] tuple:listOfTupleResult){
		System.out.print("( ");
	 	for(String t:tuple){
	 		System.out.print(t+", ");
	 	}
	 	System.out.print(") ");
	 	System.out.println();
	 	
	 	}
		
	}
	
}

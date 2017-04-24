import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.List;
import java.util.Arrays;
import java.util.List;

import parsers.ParseResultsForWS;
import parsers.WebServiceDescription;
import download.WebService;


@SuppressWarnings("unused")
public class Main {

	public static final void main(String[] args) throws Exception{
		
		//Testing without loading the description of the WS
		/*List<String> params = Arrays.asList("http://musicbrainz.org/ws/1/artist/?name=", null);
	    WebService ws=new WebService("mb_getArtistInfoByName",params);	
	    String fileWithCallResult = ws.getCallResult("Frank Sinatra");
		System.out.println("The call is   **"+fileWithCallResult+"**");
		String fileWithTransfResults=ws.getTransformationResult(fileWithCallResult);
		System.out.println("!"+fileWithTransfResults);
		ArrayList<String[]>  listOfTupleResult= ParseResultsForWS.showResults(fileWithTransfResults, ws); //won't work because ws.headvars = null
		*/
		String query = "mb_getArtistInfoByName(\"Jane Birkin\", ?id, ?b, ?e)#mb_getReleasesByArtistId(?id,\"B0006IGGOK\", ?title, ?date)#mb_getReleasesByArtist(?id,\"B0006IGGOK\", ?title, ?date)";
		ExecutionEngine engine = new ExecutionEngine();
		engine.process(query);
		System.exit(0);
		
	}
	
}

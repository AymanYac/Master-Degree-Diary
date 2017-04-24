import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map.Entry;



import java.util.TreeMap;

import parsers.ParseResultsForWS;
import parsers.WebServiceDescription;
import download.WebService;
public class ExecutionEngine {
	
	TreeMap<String,HashMap<WebService,List<String>>> BoundMaps = new TreeMap <String,HashMap<WebService,List<String>>>(); //will contain bound Vars -> WS for joins of future ws's
	private ArrayList<String[]> incrementalPartialResults = new ArrayList<String[]>();
	private HashMap<String,Integer> WSOffset = new HashMap<String,Integer>();
	List<WebService> wss = new ArrayList<>();

	public void process(String query) throws Exception {
		String[] wsstring = query.split("#");
		for (String ws:wsstring){ //Fills a WebService array following the description of each #step
			wss.add(WebServiceDescription.loadDescription(ws.split("\\(")[0]));
		}
		
		//LinkedHashSet<String> BoundVars = new LinkedHashSet<String>(); //will contain bound variables, the set is filled step by step
		
		for (WebService ws:wss){
			System.out.println("Current WS > "+ws.name);
			Boolean readyToCallWs=true; //true if a composition call's variables are computed (joins made or constant in-out's)
			LinkedHashMap<String,ArrayList<String>> JoinKeys = new LinkedHashMap<String,ArrayList<String>>(); //Collection of previous ws's columns -in order of put call- to be joined to obtain current ws inputs
			String [] params = wsstring[wss.indexOf(ws)].split("\\(")[1].replace(")", "").split(","); //List of parameters of current ws step
			List<String> inputs = new ArrayList<>(); //List of comma separated inputs for the current ws as defined in ws.getCallResult(bool,inputs)
			int i=0;
			
			for(String var:ws.headVariables){
				if(!ws.headVariableToType.get(var)){ //the variable is an output which means it's bound for future ws steps
					//BoundVars.add(var.trim());
					HashMap tmp = new HashMap<WebService,List<String>>();
					tmp.put(ws, inputs);
					BoundMaps.put(var.trim(), tmp);
				}else{
						if(!BoundMaps.keySet().contains(var) && params[i].trim().startsWith("?")){ //the variable is unbound and not a constant
							System.err.println("Workflow error : Variable "+var+" of webservice "+ws.name+" unbound");
							System.exit(1);
						}
						if(!params[i].trim().startsWith("?")){ //the variable is a constant, it's added to inputs
							inputs.add(params[i].trim().replace("\"", ""));
							ArrayList<String> JoinKey = new ArrayList<String>(); //We extract the bound variable to be used in current ws call
							JoinKey.add(params[i].trim().replace("\"", ""));
							JoinKeys.put(var, JoinKey);
						}else{ //the variable is bound to previous ws's, a join is needed
							readyToCallWs=false;
							System.out.println("Need join on "+var);
							WebService JoinWS = getWSofVar(var);
							System.out.println("Join WS > "+JoinWS.name);
							boolean cash_overwrite;
							boolean seek_join=true; //true if the dependency ws has no inputs, i.e: it has been merged after iterative calls
							if(getWSInputsofVar(var).toArray(new String[0]) == null){
								seek_join=true;
							}
							String JoinfileWithCallResult = JoinWS.getCallResult(cash_overwrite=false,seek_join,getWSInputsofVar(var).toArray(new String[0])); //we call the needed previous ws with it's corresponding inputs
							String JoinfileWithTransfResults=JoinWS.getTransformationResult(JoinfileWithCallResult); //we transform it to the standardize format
							ArrayList<String[]>  JoinlistOfTupleResult= ParseResultsForWS.showResults(JoinfileWithTransfResults, JoinWS);
							ArrayList<String> JoinKey = new ArrayList<String>(); //We extract the bound variable to be used in current ws call
							for(String [] tuple:JoinlistOfTupleResult){
								if(JoinfileWithTransfResults.matches(".*JOIN\\.xml")){
									JoinKey.add(tuple[i]);
								}else{
									JoinKey.add(tuple[i+1]);
								}
								}
							JoinKeys.put(var, JoinKey);
							
							
						}
					}
				i++;
				}
			String MergedFileWithCallResult=null;
			if(!readyToCallWs){
				String partialFileWithCallResult=null;
				Integer partialCallNumber =0;
				for(ArrayList<String> JoinKey:JoinKeys.values()){
					if(JoinKey.size() > partialCallNumber) partialCallNumber = JoinKey.size();
				}
			for(i=0;i<partialCallNumber;i++){
				List<String> tmp = new ArrayList<String>();
				for(String JKey:JoinKeys.keySet()){			//Loop through the join columns
					if(JoinKeys.get(JKey).size() > i){
						tmp.add(JoinKeys.get(JKey).get(i));		//to compose input of call
					}else{
						tmp.add(JoinKeys.get(JKey).get(0));	
					}
				}
				//String [] newInputs = (String[])ArrayUtils.addAll(tmp.toArray(new String[0]),inputs.toArray(new String[0]));
				String [] newInputs = tmp.toArray(new String[0]);
				partialFileWithCallResult = ws.getCallResult(false,false,newInputs);
			}
			XmlMerger merger = new XmlMerger();
			MergedFileWithCallResult = merger.merge(partialFileWithCallResult,ws.mergeTags.split(","));
			}else{
				MergedFileWithCallResult = ws.getCallResult(false,false,inputs.toArray(new String[0]));
			}
			appendPartial(MergedFileWithCallResult,ws);
			/*
			
			
			
			
			//Let's now make ready the ws's calls, i.e: We create the inputs variable
			//2 cases : -One join column, calls will be made for each row
			//			-2+ join columns, calls will be made for every arrangement of rows
			LinkedHashMap<String,HashSet<String>> JoinKeysIterator =JoinKeys; //copy of JoinKeys used to create arrangements
			String partialFileWithCallResult = null;
			for(String Joinkey:JoinKeys.keySet()){
				if(JoinKeys.keySet().size()>1){ //We have 2+ columns
					for(String JoinkeyIterator:JoinKeysIterator.keySet()){
						JoinKeysIterator.remove(Joinkey);
						if(JoinkeyIterator!=Joinkey){
							for(String JoinRow:JoinKeys.get(Joinkey)){
								for(String JoinRowIterator:JoinKeysIterator.get(JoinkeyIterator)){
									System.out.println(JoinRow+" "+JoinRowIterator);
									//call current ws with [join rows] instance
									partialFileWithCallResult = ws.getCallResult(false, String.join(",",inputs)+","+JoinRow+","+JoinRowIterator);
									//transform it to parsable format
									String partialFileWithTransfResults=ws.getTransformationResult(partialFileWithCallResult);
									//append it to incremental partial results
									appendPartial(partialFileWithTransfResults,ws,getWSofVar(var));
									
								}
							}
							
						}
						
					}
				}else{
					for(String JoinRow:JoinKeys.get(Joinkey)){
						//System.out.println(JoinRow);
						fileWithCallResult = ws.getCallResult(false, String.join(",",inputs)+","+JoinRow);
						//System.out.println("The call is   ##"+fileWithCallResult+"##");
					}
				}
			}
			if(fileWithCallResult!=null){
				XmlMerger merger = new XmlMerger();
				merger.merge(fileWithCallResult,"metadata","release-list");
			}else{
				boolean cash_overwrite=false;
				//System.out.println(inputs);
				fileWithCallResult = ws.getCallResult(cash_overwrite, String.join(", ", inputs));
				//System.out.println("The call is   ##"+fileWithCallResult+"##");
				String fileWithTransfResults=ws.getTransformationResult(fileWithCallResult);
				ArrayList<String[]>  listOfTupleResult= ParseResultsForWS.showResults(fileWithTransfResults, ws);
				System.out.println("The tuple results are for "+ws.name); */
				/*for(String [] tuple:listOfTupleResult){
					System.out.print("( ");
					for(String t:tuple){
						System.out.print(t+", ");
					}
					System.out.print(") ");
					System.out.println();
				}*/

			}
			
				
		}
	//}

	private WebService getWSofVar(String var) {
		Entry<WebService, List<String>> entry=BoundMaps.get(var).entrySet().iterator().next(); //entry contains set of WS's were var is bound
		return  entry.getKey(); 												   //we choose the first match (sufficient)
	
	}
	private  List<String> getWSInputsofVar(String var) {
		Entry<WebService, List<String>> entry=BoundMaps.get(var).entrySet().iterator().next(); //entry contains set of WS's were var is bound
		return entry.getValue();												   //we choose the first match (sufficient)
		
	}

	private void appendPartial(String MergedFileWithCallResult, WebService ws) throws Exception {
		ArrayList<String[]>  newPartialListOfTupleResult= ParseResultsForWS.showResults(MergedFileWithCallResult, ws);
		
	}
}

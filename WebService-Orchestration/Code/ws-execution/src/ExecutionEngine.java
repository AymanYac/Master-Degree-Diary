import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;

import parsers.ParseResultsForWS;
import parsers.WebServiceDescription;
import download.WebService;
public class ExecutionEngine {
	
	TreeMap<String,HashMap<WebService,List<String>>> BoundMaps = new TreeMap <String,HashMap<WebService,List<String>>>(); //will contain bound Vars -> WS for joins of future ws's
	private ArrayList<String[]> incrementalPartialResults = new ArrayList<String[]>();
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
							boolean seek_join;
							String JoinfileWithCallResult = JoinWS.getCallResult(cash_overwrite=false,seek_join=true,getWSInputsofVar(var).toArray(new String[0])); //we call the needed previous ws with it's corresponding inputs
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
				String [] JoinKeysArray = JoinKeys.keySet().toArray(new String[0]);
				for(String row:JoinKeys.get(JoinKeysArray[0])){
					partialFileWithCallResult = makeWSCall(ws,row,0,JoinKeysArray,JoinKeys);
				}
				
				XmlMerger merger = new XmlMerger();
				MergedFileWithCallResult = merger.merge(partialFileWithCallResult,ws.mergeTags.split(","));
			}else{
				MergedFileWithCallResult = ws.getCallResult(false,false,inputs.toArray(new String[0]));
			}
			joinPartial(MergedFileWithCallResult,ws);

			}
			
				
		}
	//}

	private String makeWSCall(WebService ws, String row, int index, String[] joinKeysArray, LinkedHashMap<String, ArrayList<String>> joinKeys) {
		if(index+1<joinKeysArray.length){
			for(String row2:joinKeys.get(joinKeysArray[index+1])){
				return makeWSCall(ws,row+","+row2,index+1,joinKeysArray,joinKeys);
			}
		}else{
			String [] newInputs = row.split(",");
			return ws.getCallResult(false,false,newInputs);
			
		}
		return null; //Syntax sugar
		
	}

	private WebService getWSofVar(String var) {
		Entry<WebService, List<String>> entry=BoundMaps.get(var).entrySet().iterator().next(); //entry contains set of WS's were var is bound
		return  entry.getKey(); 												   //we choose the first match (sufficient)
	
	}
	private  List<String> getWSInputsofVar(String var) {
		Entry<WebService, List<String>> entry=BoundMaps.get(var).entrySet().iterator().next(); //entry contains set of WS's were var is bound
		return entry.getValue();												   //we choose the first match (sufficient)
		
	}

	private void joinPartial(String MergedFileWithCallResult, WebService ws) throws Exception {
		ArrayList<String[]>  newPartialListOfTupleResult= ParseResultsForWS.showResults(MergedFileWithCallResult, ws);
		
	}
}

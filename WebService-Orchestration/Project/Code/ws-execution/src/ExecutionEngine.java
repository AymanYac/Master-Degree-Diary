import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.TreeMap;

import parsers.ParseResultsForWS;
import parsers.WebServiceDescription;
import download.WebService;
public class ExecutionEngine {
	
	TreeMap<String,HashMap<WebService,List<String>>> BoundMaps = new TreeMap <String,HashMap<WebService,List<String>>>(); //will contain bound Vars -> WS for joins of future ws's
	public ArrayList<String[]> incrementalPartialResults = new ArrayList<String[]>();
	List<WebService> wss = new ArrayList<>();
	HashSet<String> targetFiles = new HashSet<String>(); 

	public void process(String query) throws Exception {
		String[] wsstring = query.split("#");
		for (String ws:wsstring){ //Fills a WebService array following the description of each #step
			wss.add(WebServiceDescription.loadDescription(ws.split("\\(")[0]));
		}
		
		//LinkedHashSet<String> BoundVars = new LinkedHashSet<String>(); //will contain bound variables, the set is filled step by step
		
		for (WebService ws:wss){
			System.out.println("\n"+"Current WS > "+ws.name);
			Boolean readyToCallWs=true; //true if a composition call's variables are computed (joins made or constant in-out's)
			LinkedHashMap<String,ArrayList<String>> JoinKeys = new LinkedHashMap<String,ArrayList<String>>(); //Collection of previous ws's columns -in order of put call- to be joined to obtain current ws inputs
			String [] params = wsstring[wss.indexOf(ws)].split("\\(")[1].replace(")", "").split(","); //List of parameters of current ws step
			List<String> inputs = new ArrayList<>(); //List of comma separated inputs for the current ws as defined in ws.getCallResult(bool,inputs)
			int i=0;
			
			//Getting the data
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
							System.out.println("\t"+"Need join on "+var);
							WebService JoinWS = getWSofVar(var);
							System.out.println("Join WS > "+JoinWS.name);
							boolean cash_overwrite;
							boolean seek_join;
							String JoinfileWithCallResult = JoinWS.getCallResult(cash_overwrite=false,seek_join=true,getWSInputsofVar(var).toArray(new String[0])); //we call the needed previous ws with it's corresponding inputs
							String JoinfileWithTransfResults=JoinWS.getTransformationResult(JoinfileWithCallResult); //we transform it to the standardize format
							ArrayList<String[]>  JoinlistOfTupleResult= ParseResultsForWS.showResults(JoinfileWithTransfResults, JoinWS);
							ArrayList<String> JoinKey = new ArrayList<String>(); //We extract the bound variable to be used in current ws call
							for(String [] tuple:JoinlistOfTupleResult){
								//System.out.println(String.join(",", tuple));
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
			//Joining the data
			String MergedFileWithCallResult=null;
			if(!readyToCallWs){
				targetFiles.clear();
				String [] JoinKeysArray = JoinKeys.keySet().toArray(new String[0]);
				int start_index =0;
				for(String row:JoinKeys.get(JoinKeysArray[start_index])){
					makeWSCall(ws,row,start_index,JoinKeysArray,JoinKeys);
				}
				
				XmlMerger merger = new XmlMerger();
				MergedFileWithCallResult = merger.merge(targetFiles,ws.mergeTags.split(","));
			}else{
				MergedFileWithCallResult = ws.getCallResult(false,false,inputs.toArray(new String[0]));
			}
			String MergedfileWithTransfResults=ws.getTransformationResult(MergedFileWithCallResult);
			ArrayList<String[]>  MergedlistOfTupleResult= ParseResultsForWS.showResults(MergedfileWithTransfResults, ws);
			joinPartial(MergedlistOfTupleResult,ws);
			//System.out.println(String.join(",",incrementalPartialResults.get(0)));
			}
		System.out.println("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::");	
		System.out.println("::::RESULTS::::FOR::::"+query+":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::");
		System.out.println("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::");
		HashSet<String> unique_columns = new HashSet<String>();
		HashSet<Integer> unique_index = new HashSet<Integer>();
		for(int col_index=0;col_index<incrementalPartialResults.size();col_index++){
			if(!unique_columns.contains(incrementalPartialResults.get(col_index)[0])){
				unique_columns.add(incrementalPartialResults.get(col_index)[0]);
				unique_index.add(col_index);
			}
		}
		for(int row_index=0;row_index<incrementalPartialResults.get(0).length;row_index++){
			for(int col_index=0;col_index<incrementalPartialResults.size();col_index++){
				if(unique_index.contains(col_index)){
					System.out.printf("%-38.38s",incrementalPartialResults.get(col_index)[row_index]);
				}else{
					continue;
				}
		}
			System.out.println("");
		}
	}

	private void makeWSCall(WebService ws, String row, int index, String[] joinKeysArray, LinkedHashMap<String, ArrayList<String>> joinKeys) throws InterruptedException {
		if(index+1<joinKeysArray.length){
			for(String row2:joinKeys.get(joinKeysArray[index+1])){
				makeWSCall(ws,row+","+row2,index+1,joinKeysArray,joinKeys);
			}
		}else{
			if(row.contains(",,")){ //one field is missing for call
				
			}else{
				String [] newInputs = row.split(",");
				targetFiles.add(ws.getCallResult(false,false,newInputs));
			}
			
			
		}
		
	}

	private WebService getWSofVar(String var) {
		Entry<WebService, List<String>> entry=BoundMaps.get(var).entrySet().iterator().next(); //entry contains set of WS's were var is bound
		return  entry.getKey(); 												   //we choose the first match (sufficient)
	
	}
	private  List<String> getWSInputsofVar(String var) {
		Entry<WebService, List<String>> entry=BoundMaps.get(var).entrySet().iterator().next(); //entry contains set of WS's were var is bound
		return entry.getValue();												   //we choose the first match (sufficient)
		
	}

	private void joinPartial(ArrayList<String[]> MergedlistOfTupleResult, WebService ws) throws Exception {

			boolean emptyResult=true;
			if(incrementalPartialResults.size()>0){
				ArrayList<String[]> newIncrementalPartialResults = new ArrayList<String[]>();
				ArrayList<String> column_names = new ArrayList<String>();
				ArrayList<Integer> join_column_idx = new ArrayList<Integer>();
				ArrayList<Integer> join_heads_idx = new ArrayList<Integer>();
				for(int i=0;i<incrementalPartialResults.size();i++){
					String column_name = incrementalPartialResults.get(i)[0];
					//System.out.print("header: "+column_name+" ");
					column_names.add(column_name);
					newIncrementalPartialResults.add(new String[]{column_name});
					if(ws.headVariables.contains(column_name)){
						//System.out.println("hit");
						join_column_idx.add(i);
						join_heads_idx.add(ws.headVariables.indexOf(column_name));
					}else{
						//System.out.println("miss");
					}
				}
				//add new headvars as column names to incrementalPartialResults
				for(String headvar:ws.headVariables){
					incrementalPartialResults.add(new String[]{headvar});
					newIncrementalPartialResults.add(new String[]{headvar});
				}
				for(int i=1;i<incrementalPartialResults.get(0).length;i++){
					ArrayList<ArrayList<String>> valid_records = null;
					HashSet<Integer> row_range_search = new HashSet<Integer>();
					//we will look through matches in all the tuples of MergedlistOfTupleResult
					for(int j=0;j<MergedlistOfTupleResult.size();j++){
						row_range_search.add(j);
					}
					
					valid_records = joinMatch(0,row_range_search,join_column_idx,join_heads_idx,MergedlistOfTupleResult,i);
					
					if(valid_records != null){
						emptyResult=false;
						for(ArrayList<String> valid_record:valid_records){
							if(!valid_record.isEmpty()){
								for(int k=0;k<valid_record.size();k++){String [] tmp = newIncrementalPartialResults.get(k);
								ArrayList<String> tmp2 = new ArrayList<String>();
								tmp2.addAll(Arrays.asList(tmp));
								tmp2.add(valid_record.get(k));
								newIncrementalPartialResults.set(k, tmp2.toArray(new String[0]));
							}
							}
						}
					}
					
				}
			if(emptyResult) emptyResult();
			incrementalPartialResults=newIncrementalPartialResults;
			}else{
				//incrementalPartialResults is empty, i.e : this is the first filling using MergedlistOfTupleResult of first WS
				if(MergedlistOfTupleResult.isEmpty()){
					emptyResult();
				}
				//We begin by filling the column names
				for(int i=0;i<MergedlistOfTupleResult.get(0).length;i++){
					String headvar = ws.headVariables.get(i);
					incrementalPartialResults.add(new String[]{headvar});
				}
				//then we fill the body with the data for the tuples of MergedlistOfTupleResult
				for(String[] resultrow:MergedlistOfTupleResult){
					for(int j=0;j<resultrow.length;j++){
						String elem = resultrow[j];
						String [] tmp = incrementalPartialResults.get(j);
						ArrayList<String> tmp2 = new ArrayList<String>();
						tmp2.addAll(Arrays.asList(tmp));
						tmp2.add(elem);
						incrementalPartialResults.set(j, tmp2.toArray(new String[0]));
					}
				}
			}
		}

	private ArrayList<ArrayList<String>> joinMatch(int j, HashSet<Integer> old_row_range_search, ArrayList<Integer> join_column_idx,
			ArrayList<Integer> join_heads_idx, ArrayList<String[]> MergedlistOfTupleResult,int i) {
		if(j+1<join_column_idx.size()){
			HashSet<Integer> new_row_range_search = new HashSet<Integer>();
			for(Integer idx:old_row_range_search){
				String row = MergedlistOfTupleResult.get(idx)[join_heads_idx.get(j)];
				if (row.equals(incrementalPartialResults.get(join_column_idx.get(j))[i])){
					new_row_range_search.add(idx);
					
				}
			}
			if(!new_row_range_search.isEmpty()){
				//System.out.println("new range search "+new_row_range_search);
				return joinMatch(j+1,new_row_range_search,join_column_idx,join_heads_idx,MergedlistOfTupleResult,i); 
			}else{
				return null;
			}
		}else{
			ArrayList<ArrayList<String>> tmp_rows = new ArrayList<ArrayList<String>>();
			for(Integer idx:old_row_range_search){
				ArrayList<String> tmp_row = new ArrayList<String>();
				String row = MergedlistOfTupleResult.get(idx)[join_heads_idx.get(j)];
				if (row.equals(incrementalPartialResults.get(join_column_idx.get(j))[i])){
					//System.out.println(String.join(",", MergedlistOfTupleResult.get(idx)));
					for(int k=0;k < incrementalPartialResults.size();k++){
						String[] increm_col = incrementalPartialResults.get(k);
						if(increm_col.length>i){
							tmp_row.add(increm_col[i]);
							//System.out.println("!!!!!"+increm_col[i]);
						}
					}
					for(String new_row: MergedlistOfTupleResult.get(idx)){
						tmp_row.add(new_row);
						//System.out.println("!!!!!"+new_row);
					}
					
				}
				tmp_rows.add(tmp_row);
			}
			if(!tmp_rows.isEmpty()){
				/*for(ArrayList<String> row:tmp_rows){
					if(!row.isEmpty()) System.out.println(String.join(",", row));
				}*/
				return tmp_rows;
			}else{
				return null;
			}
		}
	}

	private void emptyResult() {
		System.out.println("::::::::::::::::::::::::::::::::::::::::::");
		System.out.println("No records matching query");
		System.exit(0);
	}
}

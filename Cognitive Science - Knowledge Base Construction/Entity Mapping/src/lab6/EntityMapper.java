package lab6;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

/**
 * Skeleton class for a program that maps the entities from one KB to the
 * entities of another KB.
 * 
 * @author Fabian
 *
 */
public class EntityMapper {

    /**
     * Takes as input (1) one knowledge base (2) another knowledge base and (3)
     * an output file.
     * 
     * Writes into the output file "entity1 TAB entity2 NEWLINE", if the first
     * entity from the first knowledge base is the same as the second entity
     * from the second knowledge base. Output 0 or 1 line per entity1.
     */
	//Levenshtein distance implementation from http://rosettacode.org/wiki/Levenshtein_distance#Java
	//@author :https://en.wikipedia.org/wiki/Special:Contributions/189.51.209.166
    public static int lev_dist(String a, String b) {
        int [] costs = new int [b.length() + 1];
        for (int j = 0; j < costs.length; j++)
            costs[j] = j;
        for (int i = 1; i <= a.length(); i++) {
            costs[0] = i;
            int nw = i - 1;
            for (int j = 1; j <= b.length(); j++) {
                int cj = Math.min(1 + Math.min(costs[j], costs[j - 1]), a.charAt(i - 1) == b.charAt(j - 1) ? nw : nw + 1);
                nw = costs[j];
                costs[j] = cj;
            }
        }
        return costs[b.length()];
    }
	public static boolean HasSpecialHomonymLabels(KnowledgeBase kb){
		HashSet<String> relationSetWithHomonyms = new HashSet<String>();
        HashSet<String> relationSetWithoutHomonyms = new HashSet<String>();
        HashSet<String> knownlabels = new HashSet<String>();
        for(String entity1:kb.facts.keySet()){
        	boolean hashomonym=false;
        	if(knownlabels.contains(kb.facts.get(entity1).get("rdfs:label").iterator().next())){
        		hashomonym=true;
        	}
        	knownlabels.add(kb.facts.get(entity1).get("rdfs:label").iterator().next());
        	for (String relation : kb.facts.get(entity1).keySet()){
        		relationSetWithHomonyms.add(relation);
        		if(!hashomonym) relationSetWithoutHomonyms.add(relation);
        	}
        }
        relationSetWithHomonyms.removeAll(relationSetWithoutHomonyms);
        System.out.println(kb.toString()+" "+relationSetWithHomonyms);
        return(!relationSetWithHomonyms.isEmpty());
	}
	public static void matchrelations(KnowledgeBase kb1,KnowledgeBase kb2){
		HashSet<String> entitySet1 = new HashSet<String>();
		HashSet<String> relationSet1 = new HashSet<String>();
		HashSet<String> entitySet2 = new HashSet<String>();
		HashSet<String> relationSet2 = new HashSet<String>();
        HashSet<String> knownlabels1 = new HashSet<String>();
        HashSet<String> knownlabels2 = new HashSet<String>();
		for(String entity1:kb1.facts.keySet()){
			if(!knownlabels1.contains(kb1.facts.get(entity1).get("rdfs:label").iterator().next())){
        		entitySet1.add(entity1);
        		relationSet1.addAll(kb1.facts.keySet());
        		knownlabels1.add(kb1.facts.get(entity1).get("rdfs:label").iterator().next());
        		
        	}else{
        		
        	}
		}
		
		for(String entity2:kb2.facts.keySet()){
			if(!knownlabels2.contains(kb2.facts.get(entity2).get("rdfs:label").iterator().next())){
        		entitySet1.add(entity2);
        		relationSet2.addAll(kb2.facts.keySet());
        		knownlabels2.add(kb2.facts.get(entity2).get("rdfs:label").iterator().next());
        	}else{
        		
        	}
		}
		//System.out.println(relationSet1.size());
		//System.out.println(relationSet2.size());
	}
    public static void main(String[] args) throws IOException {
        // Uncomment for your convenience. Comment it again before submission!
       /*
          args = new String[] {
          "/home/moriarty/workspace/lab6/src/lab6/yago-anonymous.tsv",
          "/home/moriarty/workspace/lab6/src/lab6/dbpedia.tsv",
          "/home/moriarty/workspace/lab6/src/lab6/result.tsv" };
       */
        KnowledgeBase kb1 = new KnowledgeBase(new File(args[0]));	//second param has no incidence, used for debugging purposes
        KnowledgeBase kb2 = new KnowledgeBase(new File(args[1]));	//can be null
        //matchrelations(kb1,kb2);
        try (Writer result = new OutputStreamWriter(new FileOutputStream(args[2]), "UTF-8")) {
            for (String entity1 : kb1.facts.keySet()) {
                String mostLikelyCandidate = null;
                int distance=99999;
                for (String entity2 : kb2.facts.keySet()) {
                	int label_distance=999;
                    for(String label1:kb1.facts.get(entity1).get("rdfs:label")){
                    	for(String label2:kb2.facts.get(entity2).get("rdfs:label")){
                    		if(lev_dist(label1,label2)<label_distance) label_distance=lev_dist(label1,label2);
                    	}
                    	
                    }
                    int relation_distance=999;
                    for(String relation1:kb1.facts.get(entity1).keySet()){
                    	if(!relation1.equals("rdfs:label")){
	                    	for(String y1:kb1.facts.get(entity1).get(relation1)){
	                    		for(String relation2:kb2.facts.get(entity2).keySet()){
	                    			if(!relation2.equals("rdfs:label")){
		                    			for(String y2:kb2.facts.get(entity2).get(relation2)){
		                    				if(lev_dist(y1,y2)<relation_distance) relation_distance=lev_dist(y1,y2);
		                    		}
	                    		}
	                    		}
	                    	}
                    	}
                    }
                    if(label_distance+5*relation_distance<distance && label_distance+5*relation_distance<62){
                    	mostLikelyCandidate=entity2;
                    	distance=label_distance+5*relation_distance;
                    	
                    }

                }
                if (mostLikelyCandidate != null) {
                    result.write(entity1 + "\t" + mostLikelyCandidate + "\t" + distance + "\n");
                }
                kb2.facts.remove(mostLikelyCandidate);
            }
        }
    }
}
package lab4;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Skeleton for a Viterbi POS tagger.
 * 
 * @author Fabian M. Suchanek
 *
 */
public class Viterbi {

    /** HMM we'll use */
    protected HiddenMarkovModel model;

    /** Constructs the parser from a model file */
    public Viterbi(File modelFile) throws FileNotFoundException, IOException {
        model = new HiddenMarkovModel(modelFile);
    }

    /** Parses a sentence and returns the list of POS tags */
    public List<String> parse(String sentence) {
        List<String> words = Arrays.asList((". " + sentence.toLowerCase() + " .").split(" "));
        Map<Integer, Map<String,Double>> nodeProb = new TreeMap <Integer,Map<String,Double>>();
        Map<Integer, Map<String,String>> nodePath = new TreeMap <Integer,Map<String,String>>();
        Map<Integer,Map<String,Double>> bestScore= new TreeMap<Integer, Map<String,Double>>();
        List<String> returned = new ArrayList<String>();
        for(String word:words){
        	if(word.equals(".")){
        		if(nodeProb.isEmpty()){
        			//Start of sentence
        			Map<String,Double> tmp = new HashMap<String,Double>();
        			tmp.put("STRT", 1.0);
        			nodeProb.put(0, tmp);
        			Map<String,String> tmp2 = new HashMap<String,String>();
        			tmp2.put("STRT", "STRT");
        			nodePath.put(0, tmp2);
        		}else{
        			//End of sentence
        			int stage=words.size()-1;
        			String output=null;
        			Map<String,Double> tmp = new HashMap<String,Double>();
        			tmp.put(".", -1.0);
        			bestScore.put(stage, tmp);
	        			for(String fromTag:nodeProb.get(stage-1).keySet()){
	        				double fromTagscore=nodeProb.get(stage-1).get(fromTag)*model.getTransitionProbability(fromTag, ".")*model.getEmissionProbability(".", ".");
	        				if(fromTagscore>bestScore.get(stage).get(".")){
	        					output=nodePath.get(stage-1).get(fromTag)+" .";
	        					bestScore.get(stage).replace(".", fromTagscore);
	        				}
	        			}
	        			
        			returned.addAll(Arrays.asList(output.split(" ")));
        		}
        	}else{
        			//Word of sequence
        			int stage=words.indexOf(word);
        			boolean task4=false;//used to check if word belongs to know words ( task4 )
        			
        			for(String toTag:model.emissionProb.keySet()){
        				if(bestScore.containsKey(stage)){
        					bestScore.get(stage).put(toTag, -1.0);
        				}else{
        					bestScore.put(stage, new HashMap(){{put(toTag,-1.0);}});
        				}
        				
        				for(String fromTag: nodeProb.get(stage-1).keySet()){
	        					double fromTagscore=nodeProb.get(stage-1).get(fromTag)*model.getTransitionProbability(fromTag, toTag)*model.getEmissionProbability(toTag, word);
	        					if(fromTagscore>0 && fromTagscore>bestScore.get(stage).get(toTag)){
	        						task4=true;
	        						if(nodeProb.containsKey(stage)){
	        							nodeProb.get(stage).put(toTag, fromTagscore);
	        							nodePath.get(stage).put(toTag, nodePath.get(stage-1).get(fromTag)+" "+toTag);
	        						}else{
	        							nodeProb.put(stage, new HashMap(){{put(toTag,fromTagscore);}});
	        							nodePath.put(stage, new HashMap(){{put(toTag,nodePath.get(stage-1).get(fromTag)+" "+toTag);}});
	        						}
	        						bestScore.get(stage).put(toTag, fromTagscore);
	        					}
        					}
        				
        			}
        			if(!task4){
        				for(String toTag:model.emissionProb.keySet()){
            				if(bestScore.containsKey(stage)){
            					bestScore.get(stage).put(toTag, -1.0);
            				}else{
            					bestScore.put(stage, new HashMap(){{put(toTag,-1.0);}});
            				}
            				
            				for(String fromTag: nodeProb.get(stage-1).keySet()){
    	        					double fromTagscore=nodeProb.get(stage-1).get(fromTag)*model.getTransitionProbability(fromTag, toTag);
    	        					if(fromTagscore>0 && fromTagscore>bestScore.get(stage).get(toTag)){
    	        						if(nodeProb.containsKey(stage)){
    	        							nodeProb.get(stage).put(toTag, fromTagscore);
    	        							nodePath.get(stage).put(toTag, nodePath.get(stage-1).get(fromTag)+" "+toTag);
    	        						}else{
    	        							nodeProb.put(stage, new HashMap(){{put(toTag,fromTagscore);}});
    	        							nodePath.put(stage, new HashMap(){{put(toTag,nodePath.get(stage-1).get(fromTag)+" "+toTag);}});
    	        						}
    	        						bestScore.get(stage).put(toTag, fromTagscore);
    	        					}
            					}
            				
            			}
        			}
        		
        	}
        }
        
        // Smart things happen here!
        return (returned);
    }

    /**
     * Given (1) a Hidden Markov Model file and (2) a sentence (in quotes),
     * prints the sequence of POS tags
     */
    public static void main(String[] args) throws Exception {
    	System.out.println(new Viterbi(new File(args[0])).parse(args[1]));
    	//task 4
    	System.out.println("Tagging for 'Elvis is in Krzdgwzy': "+new Viterbi(new File(args[0])).parse("Elvis is in Krzdgwzy"));
    	System.out.println("Tagging for 'Elvis is in Krzdgwzy jdieja': "+new Viterbi(new File(args[0])).parse("Elvis is in Krzdgwzy jdieja"));
    	//task 3, uncomment below
    	/*System.out.println("___________________");
    	System.out.println("Tagging for 'Elvis is the best': "+new Viterbi(new File(args[0])).parse("Elvis is the best"));
    	System.out.println("This means that 'elvis' is recognized as a proper noun , 'is' as a 3rd person verb, 'the' as determinant \n"+
    	"and best as a superlative adjective. This means that elvis is 'the most good' with 'good' as an adjective ");
    	System.out.println("___________________");
    	System.out.println("Tagging for 'Elvis sings best': "+new Viterbi(new File(args[0])).parse("Elvis sings best"));
    	System.out.println("This means that 'elvis' is recognized as a proper noun , 'sings' as a 3rd person verb \n"+
    	    	"and 'best' as an adverb. The sentence can be understood as the answer to the question 'How does elvis sing?' ");
    	System.out.println("___________________");
    	System.out.println("The results are different because:\n"
    			+ "Prob(DT,JJS,.|'is the best') = 0.000004475 since \nT\tDT\tJJS\t0.0071043431285610905\nE\tJJS\tbest\t0.1\nT\tJJS\t.\t0.006299212598425197"
    			+ "\nWhere Prob(DT,RB,.|'is the best') = 0.000001145 since \nT\tDT\tRB\t0.004984252209277436\nE\tRB\tbest\t0.005242791162152041\nT\tRB\t.\t0.04381475471227063\n\n"
    			+ "In a similar fashion, for the 2nd sentence P(VBZ,RB,.|'sings best.') > P(VBZ,JJS,.|'sings best.)");
    	*/
    }
}
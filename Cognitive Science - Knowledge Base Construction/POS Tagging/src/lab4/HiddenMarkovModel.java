package lab4;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

/**
 * Skeleton for a Hidden Markov Model
 *
 * @author Fabian M. Suchanek
 *
 */
public class HiddenMarkovModel {

  /**
   * Stores transition probabilities of the form ADJ -> { NN -> 0.99, VBZ ->
   * 0.01 }. Should sum to 1 for each tag.
   */
  protected Map<String, Map<String, Double>> transitionProb = new TreeMap<String, Map<String, Double>>();
  protected Map<String,Double> transitionCardinality = new HashMap<String,Double>();
  /**
   * Stores emission probabilities of the form PN -> { "Elvis" -> 0.8,
   * "Priscilla" -> 0.2 }. Should sum to 1 for each tag.
   */
  protected Map<String, Map<String, Double>> emissionProb = new TreeMap<String, Map<String, Double>>();
  protected Map<String,Double> emissionCardinality = new HashMap<String,Double>();
  /** Retrieves the emission probability for this tag and this word */
  public double getEmissionProbability(String tag, String word) {
    if (!emissionProb.containsKey(tag)) return (0);
    if (!emissionProb.get(tag).containsKey(word)) return (0);
    return (emissionProb.get(tag).get(word));
  }

  /** Retrieves the transition probability for these tags */
  public double getTransitionProbability(String fromTag, String toTag) {
    if (!transitionProb.containsKey(fromTag)) return (0);
    if (!transitionProb.get(fromTag).containsKey(toTag)) return (0);
    return (transitionProb.get(fromTag).get(toTag));
  }

  /**
   * Constructs a Hidden Markov Model from a tagged Wikipedia corpus, i.e,
   * fills the fields transitionProb and emissionProb. Lowercase all words.
   */
  public HiddenMarkovModel(String wikipediaCorpus) throws IOException {
	  
    try (Parser parser = new Parser(new File(wikipediaCorpus))) {
    	Page nextPage=null;
    	String[] wordsWithTags=null;
    	String scanWord=null;
    	String scanTag=null;
    	String scanFromTag=null;
      while (parser.hasNext()) {
        nextPage = parser.next();
        scanFromTag="STRT"; //Start of sentence tag
        wordsWithTags = nextPage.content.split(" ");
        for(String token:wordsWithTags){
        	scanWord=token.split("/")[0];
        	scanTag=token.split("/")[1];
        	if(scanWord.equals("<")){
        		scanWord=token.replaceAll("/[A-Z]+$","");
        		scanTag=token.replace("</", "<").split("/")[1];
        		incrementEmissionProbability(scanTag,scanWord.toLowerCase());
            	incrementTransitionProbability(scanFromTag,scanTag);
            	scanFromTag=scanTag;
            	continue;
        	}
        	if(scanWord.matches(".*\\\\")){
        		scanWord=token.replaceAll("/[A-Z]+$","").replace("\\/", "/");
        		scanTag=token.replace("\\/", "").split("/")[1];
        	}
        	if(scanWord.equals("http:")){
        		scanTag="NNP";
        		scanWord=token.replaceAll(".NNP", "");
        	}
        	if(scanTag.matches("[1-9]")){
        		scanTag=token.split("/")[2];
        		scanWord=token.split("/")[0]+"/"+token.split("/")[1];
        	}
        	incrementEmissionProbability(scanTag,scanWord.toLowerCase());
        	incrementTransitionProbability(scanFromTag,scanTag);
        	scanFromTag=scanTag;
        }
        
      }
      normalize();//make probabilities sum to 1 for transition and emission
    }
  }

  private void normalize() {
	  Double tmp=0.0;
	  for(String tag:emissionProb.keySet()){
		  tmp=emissionCardinality.get(tag);
		  for(String word:emissionProb.get(tag).keySet()){
			  emissionProb.get(tag).replace(word, emissionProb.get(tag).get(word)/tmp);
			  
			  
		  }
	  }
	  //adds emission probability of 1 to tag STRT of word STRT (useful for concise Viterbi coding)
	  Map<String,Double> tmp2 = new HashMap<String,Double>();
	  tmp2.put("STRT", 1.0);
	  emissionProb.put("STRT", tmp2);
	  
	  for(String fromTag:transitionProb.keySet()){
		  tmp=transitionCardinality.get(fromTag);
		  for(String toTag:transitionProb.get(fromTag).keySet()){
			  transitionProb.get(fromTag).replace(toTag, transitionProb.get(fromTag).get(toTag)/tmp);
			  
		  }
	  }
	
}

private void incrementTransitionProbability(String scanFromTag, String scanTag) {
	if(transitionProb.containsKey(scanFromTag)){
		if(transitionProb.get(scanFromTag).containsKey(scanTag)){
			transitionProb.get(scanFromTag).replace(scanTag, transitionProb.get(scanFromTag).get(scanTag)+1.0);
			transitionCardinality.replace(scanFromTag, transitionCardinality.get(scanFromTag)+1.0);
		}else{
			transitionProb.get(scanFromTag).put(scanTag, 1.0);
			transitionCardinality.replace(scanFromTag, transitionCardinality.get(scanFromTag)+1.0);
		}
	}else{
		Map<String,Double> tmp = new HashMap<String,Double>();
		tmp.put(scanTag, 1.0);
		transitionProb.put(scanFromTag, tmp);
		transitionCardinality.put(scanFromTag, 1.0);
	}
	
}

private void incrementEmissionProbability(String scanTag, String scanWord) {
	if(emissionProb.containsKey(scanTag)){
		if(emissionProb.get(scanTag).containsKey(scanWord)){
			emissionProb.get(scanTag).replace(scanWord, emissionProb.get(scanTag).get(scanWord)+1.0);
			emissionCardinality.replace(scanTag, emissionCardinality.get(scanTag)+1.0);
		}else{
			emissionProb.get(scanTag).put(scanWord, 1.0);
			emissionCardinality.replace(scanTag, emissionCardinality.get(scanTag)+1.0);
		}
	}else{
		Map<String,Double> tmp = new HashMap<String,Double>();
		tmp.put(scanWord, 1.0);
		emissionProb.put(scanTag, tmp);
		emissionCardinality.put(scanTag, 1.0);
	}
	
}

/** Saves this model to a file */
  public void saveTo(File model) throws IOException {
    try (Writer out = new FileWriter(model)) {
      for (String fromTag : transitionProb.keySet()) {
        Map<String, Double> map = transitionProb.get(fromTag);
        for (String toTag : map.keySet()) {
          out.write("T\t" + fromTag + "\t" + toTag + "\t" + map.get(toTag) + "\n");
        }
      }
      for (String tag : emissionProb.keySet()) {
        Map<String, Double> map = emissionProb.get(tag);
        for (String word : map.keySet()) {
          out.write("E\t" + tag + "\t" + word + "\t" + map.get(word) + "\n");
        }
      }
    }
  }

  /**
   * Constructs a Hidden Markov Model from a previously stored model file.
   */
  public HiddenMarkovModel(File model) throws FileNotFoundException, IOException {
    try (BufferedReader in = new BufferedReader(new FileReader(model))) {
      for (String line = in.readLine(); line != null; line = in.readLine()) {
        String[] split = line.split("\t");
        if (split[0].equals("T")) {
          Map<String, Double> map = transitionProb.get(split[1]);
          if (map == null) transitionProb.put(split[1], map = new TreeMap<>());
          map.put(split[2], Double.parseDouble(split[3]));
        } else if (split[0].equals("E")) {
          Map<String, Double> map = emissionProb.get(split[1]);
          if (map == null) emissionProb.put(split[1], map = new TreeMap<>());
          map.put(split[2], Double.parseDouble(split[3]));
        }
      }
    }
  }

  /**
   * Given (1) a POS-tagged Wikipedia corpus and (2) a target model file,
   * constructs the model and stores it in the target model file.
   */
  public static void main(String[] args) throws IOException {
    HiddenMarkovModel model = new HiddenMarkovModel(args[0]);
    model.saveTo(new File(args[1]));
  }

}
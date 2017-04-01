package lab3;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
/**
 * Skeleton class to perform disambiguation
 * 
 * @author Jonathan Lajus
 *
 */
public class Disambiguation {

  /** This program takes 3 command line arguments, namely the paths to:
      - yagoLinks.tsv
      - yagoLabels.tsv
      - wikipedia-ambiguous.txt      
   in this order. You may also ignore the last argument at your will.
   The program prints statements of the form:
      <pageTitle>  TAB  <yagoEntity> NEWLINE   
   It is OK to skip articles.      
  */
  public static void main(String[] args) throws IOException {
    if (args.length < 3) {
      System.err.println("usage: Disambiguation <yagoLinks> <yagoLabels> <wikiText>");
      return;
    }
    File dblinks = new File(args[0]);
    File dblabels = new File(args[1]);
    File wiki = new File(args[2]);

    SimpleDatabase db = new SimpleDatabase(dblinks, dblabels);

    try (Parser parser = new Parser(wiki)) {
    	Page nextPage = new Page(null,null);
    	String pageTitle=null;
    	String pageContent=null;
    	String pageLabel=null;
        String correspondingYagoEntity = "<For_you_to_find>";
        
        String oldpageLabel = "";
        String entity_description=null;
        Set<String> possibleEntities = Collections.emptySet();
        HashMap<String,Integer> counter = new HashMap<String, Integer>();
        HashSet<String> h1 = new HashSet<String>(), h2 = new HashSet<String>(), finalContentSet = new HashSet<String>();
        
      while (parser.hasNext()) {
        nextPage = parser.next();
        pageTitle = nextPage.title;
        pageLabel = nextPage.label();
    	
        pageContent = nextPage.content.replace(".", "").replace(",", "").replace("(", "").replace(")", "").replace(";", "").replaceAll("[0-9]+", "").replaceAll("[0-9]+s", "");
        Set<String> ContentSet = new HashSet<String>(Arrays.asList(pageContent.split(" ")));
        // stop word list from www.ranks.nl
        // http://www.ranks.nl/stopwords
        String[] stopwords = {"a", "as", "able", "about", "above", "according", "accordingly", "across", "actually",
        		"after", "afterwards", "again", "against", "aint", "all", "allow", "allows", "almost", "alone", "along",
        		"already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody",
        		"anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate",
        		"are", "arent", "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "be",
        		"became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe",
        		"below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "cmon", "cs",
        		"came", "can", "cant", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com",
        		"come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains",
        		"corresponding", "could", "couldnt", "course", "currently", "definitely", "described", "despite", "did", "didnt",
        		"different", "do", "does", "doesnt", "doing", "dont", "done", "down", "downwards", "during", "each", "edu", "eg", 
        		"eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", 
        		"everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "far", "few", "ff",
        		"fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from",
        		"further", "furthermore", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten",
        		"greetings", "had", "hadnt", "happens", "hardly", "has", "hasnt", "have", "havent", "having", "he", "hes", "hello",
        		"help", "hence", "her", "here", "heres", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him",
        		"himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "id", "ill", "im", "ive", "ie", "if", "ignored",
        		"immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead",
        		"into", "inward", "is", "isnt", "it", "itd", "itll", "its", "its", "itself", "just", "keep", "keeps", "kept", "know",
        		"knows", "known", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked",
        		"likely", "little", "look", "looking", "looks", "ltd", "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely",
        		"might", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "nd", "near", "nearly",
        		"necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none",
        		"noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "obviously", "of", "off", "often", "oh", "ok",
        		"okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours",
        		"ourselves", "out", "outside", "over", "overall", "own", "particular", "particularly", "per", "perhaps", "placed", "please",
        		"plus", "possible", "presumably", "probably", "provides", "que", "quite", "qv", "rather", "rd", "re", "really", "reasonably",
        		"regarding", "regardless", "regards", "relatively", "respectively", "right", "said", "same", "saw", "say", "saying", "says",
        		"second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent",
        		"serious", "seriously", "seven", "several", "shall", "she", "should", "shouldnt", "since", "six", "so", "some", "somebody",
        		"somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify",
        		"specifying", "still", "sub", "such", "sup", "sure", "ts", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks",
        		"thanx", "that", "thats", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "theres", "thereafter",
        		"thereby", "therefore", "therein", "theres", "thereupon", "these", "they", "theyd", "theyll", "theyre", "theyve", "think", "third",
        		"this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too",
        		"took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "un", "under", "unfortunately", "unless",
        		"unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "value", "various", "very",
        		"via", "viz", "vs", "want", "wants", "was", "wasnt", "way", "we", "wed", "well", "were", "weve", "welcome", "well", "went", "were",
        		"werent", "what", "whats", "whatever", "when", "whence", "whenever", "where", "wheres", "whereafter", "whereas", "whereby", "wherein",
        		"whereupon", "wherever", "whether", "which", "while", "whither", "who", "whos", "whoever", "whole", "whom", "whose", "why", "will",
        		"willing", "wish", "with", "within", "without", "wont", "wonder", "would", "would", "wouldnt", "yes", "yet", "you", "youd", "youll",
        		"youre", "youve", "your", "yours", "yourself", "yourselves", "zero"};
    	Set<String> stopWordSet = new HashSet<String>(Arrays.asList(stopwords));
    	for( String word:ContentSet){
    		if(!stopWordSet.contains(word)){
    			finalContentSet.add(word);
    		}
    	}
    	pageContent = String.join(" ", finalContentSet);
    	finalContentSet.clear();
    	
        String[] content_iterable=pageContent.split(" ");
	
        if(!pageLabel.equals(oldpageLabel)){
        possibleEntities = db.reverseLabels.get(pageLabel);
        }
        for (String entity: possibleEntities){
        	entity_description = String.join(" ", db.links.get(entity)).replaceAll("wikicat|wordnet|<_|<|_>|>|,", "").replace("_", " ");
        	String[] entity_desc_iterable=entity_description.split(" ");
        	for(int i = 0; i < content_iterable.length; i++)                                            
        	{
        	  h1.add(content_iterable[i]);
        	}
        	for(int i = 0; i < entity_desc_iterable.length; i++)
        	{
        	  h2.add(entity_desc_iterable[i]);
        	}
        	h1.retainAll(h2);
        	counter.put(entity,h1.size());
        	h1.clear();
        	h2.clear();
        }
        
        correspondingYagoEntity = Collections.max(counter.entrySet(), Map.Entry.comparingByValue()).getKey(); 
        possibleEntities.remove(correspondingYagoEntity);
        counter.clear();
        System.out.println(pageTitle + "\t" + correspondingYagoEntity);
        oldpageLabel=pageLabel;
        //System.out.println(pageTitle + "\t" + db.reverseLabels.get(pageLabel) +"\t"+ db.links.get(db.reverseLabels.get(pageLabel).toArray()[1]) );
        //System.out.println(possibleEntities);
        //System.out.println(pageLabel + "\t" + db.labels.get("<"+pageLabel.replace(" ", "_")+">") + "\t" + db.links.get("<"+pageLabel.replace(" ", "_")+">") + "\t" + db.reverseLabels.get(pageLabel));
      }
    }
  }
}
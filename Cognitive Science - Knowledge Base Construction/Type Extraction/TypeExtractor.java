package lab2;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

/**
 * Skeleton code for a type extractor.
 */
public class TypeExtractor {

  /**
Given as argument a Wikipedia file, the task is to run through all Wikipedia articles,
and to extract for each article the type (=class) of which the article
entity is an instance. For example, from a page starting with "Leicester is a city",
you should extract "city". 

* extract the longest possible type ("American rock-and roll star") consisting of adjectives,
  nationalities, and nouns
* if the type cannot reasonably be extracted ("Mathematics was invented in the 19th century"),
  skip the article (do not output anything)
* take only the first item of a conjunction ("and")
* do not extract provenance ("from..", "in..", "by.."), but do extract complements
  ("body of water")
* do not extract too general words ("type of", "way", "form of"), but resolve like a
  human ("A Medusa  is one of two forms of certain animals" -> "animals")
* keep the plural

The output shall be printed to the screen in the form
    entity TAB type NEWLINE
with one or zero lines per entity.
   */
  public static void main(String args[]) throws IOException {
  //args = new String[] { "c:/fabian/data/wikipedia/wikipedia.txt" };
  //args = new String[] { "/home/moriarty/workspace/KBC_lab2/src/lab2/wikipedia-first.txt" };
  //PrintWriter writer = new PrintWriter("/home/moriarty/Desktop/DK/KBC/out.txt", "UTF-8");
  
  List<String> contentWords;
  String type;
  Page nextPage;
  Boolean flag;

  
  
    try (Parser parser = new Parser(new File(args[0]))) {
      while (parser.hasNext()) {
        nextPage = parser.next();
        flag=false;
        type=null;
        contentWords = Arrays.asList(nextPage.content.split(" "));
        for (int i =0; i<contentWords.size();i++){
          String word = contentWords.get(i);
          /*if(flag2==true && (!word.matches("(.)*st") && !word.matches("(.)*'s") && !word.matches("[0-9]+") && !(word.equals("one") || word.equals("two") || word.equals("three") || word.equals("four") || word.equals("five") || word.equals("six") || word.equals("seven") || word.equals("eight") || word.equals("nine") || word.equals("ten") || word.equals("eleven") || word.equals("twelve") || word.equals("thirteen")))){
    	  flag=true;
    	  type+=" "+word.replace(",", "");
    	  flag2=false;
      }
      
      if(word.equals("one") && i+1<contentWords.size() && contentWords.get(i+1).equals("of")){
    	  flag2=true;
      }*/
          if(word.equals("was") || word.equals("are") || word.equals("is")){
            flag=true;
            continue;
          }
          if(word.equals("and") || word.equals("between") || word.equals("in") || word.equals("for") || word.equals("nor") || word.equals("but") || word.equals("or") || word.equals("yet") || word.equals("so") || word.equals("that") || word.equals("by") || word.equals("with") || word.equals("used")){
            flag=false;
            break;
          }
          
          if(word.equals("the") || word.equals("a") || word.equals("an")){
            continue;
          }
          if(flag == true && word.matches("(.)*,")){
            type+=" "+word.replace(",", "");
            flag=false;
            break;
          }
          if(flag==true){
            type+=" "+word;
          }
          
          }
        
        if(type!=null){
        	System.out.println(nextPage.title+"\t"+type.replaceAll("null ","").replace(".", ""));
        	//writer.println(nextPage.title+"\t"+type.replaceAll("null ","").replace(".", ""));
        	
        }
      }
    }
    //writer.close();
  }

}
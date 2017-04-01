#!/bin/bash
rm -rf output/
rm CWBTIAB*.class
rm CWBTIAB.jar
hadoop com.sun.tools.javac.Main CWBTIAB.java
jar -cf CWBTIAB.jar CWBTIAB*.class
hadoop jar CWBTIAB.jar CWBTIAB input/ output
clear
cat output/part-r-00000
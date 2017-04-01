<map version="1.0.1">
<!-- To view this file, download free mind mapping software FreeMind from http://freemind.sourceforge.net -->
<node CREATED="1444034897262" ID="ID_1748406381" MODIFIED="1444115362909">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p style="text-align: center">
      Architectures for<br />Massive Data Management
    </p>
  </body>
</html></richcontent>
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1444035019280" ID="ID_415692425" MODIFIED="1475133726167" POSITION="left">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p style="text-align: center">
      extend and reuse<br />techniques from
    </p>
  </body>
</html></richcontent>
<node CREATED="1444035027025" ID="ID_397008846" MODIFIED="1444035031720" TEXT="Distributed Systems">
<node CREATED="1444035035216" ID="ID_1154277939" MODIFIED="1475133764402" TEXT="are examples of">
<node CREATED="1444035067221" ID="ID_1570310738" MODIFIED="1444035184282">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p style="text-align: center">
      Distributed file<br />systems (DFS)
    </p>
  </body>
</html></richcontent>
</node>
</node>
<node CREATED="1444035059886" ID="ID_1383317739" MODIFIED="1475133790831" TEXT="are examples of">
<node CREATED="1444035088623" ID="ID_1928686499" MODIFIED="1444035191853">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p style="text-align: center">
      Distributed operating<br />systems
    </p>
  </body>
</html></richcontent>
</node>
</node>
</node>
<node CREATED="1444034944893" ID="ID_95933114" MODIFIED="1475056512408" TEXT="Distributed Databases">
<node CREATED="1444035004883" ID="ID_465065488" MODIFIED="1475133860384" TEXT="extend techniques from">
<node CREATED="1444035008951" ID="ID_1978820339" MODIFIED="1475133861798" TEXT="(Centralized) databases">
<node CREATED="1444977894552" ID="ID_1897866367" MODIFIED="1444977960954" TEXT="in particular">
<node CREATED="1444977903224" ID="ID_1491043470" MODIFIED="1475056334919" TEXT="indexes">
<node CREATED="1444977905384" ID="ID_1861423866" MODIFIED="1444977909869" TEXT="B+ trees"/>
<node CREATED="1444977911691" ID="ID_1307865736" MODIFIED="1444977914443" TEXT="B trees"/>
<node CREATED="1444977915349" ID="ID_1876524773" MODIFIED="1444977918214" TEXT="Static hashing"/>
<node CREATED="1444977918715" ID="ID_846820059" MODIFIED="1444977928752" TEXT="Dynamic hashing">
<node CREATED="1444977930229" ID="ID_616227127" MODIFIED="1444977936799" TEXT="Extendable hashing"/>
</node>
<node CREATED="1444977937700" ID="ID_527745244" MODIFIED="1444977944519" TEXT="Spatial indexes">
<node CREATED="1444977945178" ID="ID_1023542673" MODIFIED="1444977947825" TEXT="kd-trees"/>
<node CREATED="1444977948830" ID="ID_550772735" MODIFIED="1444977950427" TEXT="R-trees"/>
<node CREATED="1444978026486" ID="ID_1761667420" MODIFIED="1444978029108" TEXT="Quad trees"/>
</node>
</node>
</node>
</node>
</node>
</node>
<node CREATED="1444038326765" ID="ID_949048109" MODIFIED="1444038349613" TEXT="Stream Data Management Systems"/>
</node>
<node CREATED="1444035221933" ID="ID_1419448354" MODIFIED="1444207552549" POSITION="right" TEXT="characterized by">
<node CREATED="1444035225820" ID="ID_905271143" MODIFIED="1475056497605" TEXT="dimensions">
<node CREATED="1444035229401" ID="ID_258853275" MODIFIED="1444035232366" TEXT="distribution scale">
<node CREATED="1444035472753" ID="ID_854284681" MODIFIED="1444035494408" TEXT="centralized"/>
<node CREATED="1444035476815" ID="ID_1536029934" MODIFIED="1444035492413" TEXT="moderate"/>
<node CREATED="1444035484308" ID="ID_103359016" MODIFIED="1444035489315" TEXT="large"/>
</node>
<node CREATED="1444035234624" ID="ID_1482692530" MODIFIED="1444035336160" TEXT="data model">
<node CREATED="1444035447386" ID="ID_794184374" MODIFIED="1444035451078" TEXT="relational"/>
<node CREATED="1444035452268" ID="ID_512836561" MODIFIED="1444035467118" TEXT="document (XML, JSON)"/>
<node CREATED="1444035455037" ID="ID_1622397651" MODIFIED="1444035458042" TEXT="key-value"/>
<node CREATED="1444035469314" ID="ID_320761053" MODIFIED="1444035471930" TEXT="graph"/>
</node>
<node CREATED="1444035239352" ID="ID_1337119329" MODIFIED="1444035519324" TEXT="data heterogeneity">
<node CREATED="1444035498013" ID="ID_1893366418" MODIFIED="1444035499520" TEXT="none"/>
<node CREATED="1444035509225" ID="ID_1309011337" MODIFIED="1444035511838" TEXT="some"/>
<node CREATED="1444035519325" ID="ID_333068872" MODIFIED="1444035521574" TEXT="a lot"/>
</node>
<node CREATED="1444035244573" ID="ID_1455871431" MODIFIED="1444035352144" TEXT="data manipulation language">
<node CREATED="1444035523168" ID="ID_604567656" MODIFIED="1444035525616" TEXT="put/get"/>
<node CREATED="1444035526509" ID="ID_1718143941" MODIFIED="1444035535245" TEXT="declarative query language"/>
</node>
<node CREATED="1444035248952" ID="ID_602940347" MODIFIED="1444035365838">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p style="text-align: center">
      distribution of<br />control and work
    </p>
  </body>
</html></richcontent>
<node CREATED="1444035541843" ID="ID_573144940" MODIFIED="1444035555237" TEXT="independent (symetric) nodes"/>
<node CREATED="1444035556346" ID="ID_1520349482" MODIFIED="1444035558919" TEXT="master-slaves"/>
<node CREATED="1444035559828" ID="ID_1497518734" MODIFIED="1444035569491" TEXT="set of masters, set of slaves"/>
</node>
<node CREATED="1444035382769" ID="ID_1643320592" MODIFIED="1444036208587" TEXT="concurrency control model">
<node CREATED="1444036178602" ID="ID_614934460" MODIFIED="1444036187989" TEXT="mostly relevant if updates!"/>
<node CREATED="1444036189436" ID="ID_1213066181" MODIFIED="1444036193493" TEXT="atomicity"/>
<node CREATED="1444036194492" ID="ID_776932557" MODIFIED="1444036201210" TEXT="partition-tolerance"/>
<node CREATED="1444036209337" ID="ID_715275066" MODIFIED="1444036212326" TEXT="consistency"/>
</node>
<node CREATED="1444036431701" ID="ID_1397300742" MODIFIED="1444036439833" TEXT="replication">
<node CREATED="1444040796043" ID="ID_406769556" MODIFIED="1444040798766" TEXT="impacts">
<node CREATED="1444040798767" ID="ID_89948995" MODIFIED="1444040800483" TEXT="consistency"/>
</node>
</node>
<node CREATED="1444038527145" ID="ID_1176884414" MODIFIED="1444038530560" TEXT="durability">
<node CREATED="1444038530560" ID="ID_1391233329" MODIFIED="1444038536832" TEXT="none (in-memory)"/>
<node CREATED="1444038538844" ID="ID_1008156604" MODIFIED="1444038547151" TEXT="some (periodic or on-demand back-up)"/>
<node CREATED="1444038548240" ID="ID_477861738" MODIFIED="1444038555429" TEXT="systematic (write-to-disk)"/>
</node>
</node>
</node>
<node CREATED="1444035413320" ID="ID_1900481012" MODIFIED="1444207567349" POSITION="right">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p style="text-align: center">
      Sample systems&#160;(mostly uncomparable,<br />some built on each other!)
    </p>
    <p style="text-align: center">
      These are just <font color="#ff0066"><b>names,</b></font>
    </p>
    <p style="text-align: center">
      check out the <font color="#0033ff"><b>dimensions</b></font>
    </p>
    <p style="text-align: center">
      characterizing each of them.
    </p>
    <p style="text-align: center">
      Many of these are also called &quot;<b>NoSQL</b>&quot;
    </p>
  </body>
</html></richcontent>
<node CREATED="1444035433051" ID="ID_660335066" MODIFIED="1475056222733" TEXT="distributed  databases"/>
<node CREATED="1444035438357" ID="ID_1617091872" MODIFIED="1444035441904" TEXT="key-value stores">
<node CREATED="1444037693585" ID="ID_729164180" MODIFIED="1444037696106" TEXT="such as">
<node CREATED="1444037696107" ID="ID_1866113944" MODIFIED="1444037698623" TEXT="Voldemort"/>
<node CREATED="1444037700278" ID="ID_1866685011" MODIFIED="1444037704091" TEXT="Redis"/>
<node CREATED="1444037705063" ID="ID_884699383" MODIFIED="1444040474284" TEXT="DynamoDB from Amazon"/>
<node CREATED="1444076883373" ID="ID_915426823" MODIFIED="1444076887539" TEXT="Oracle NoSQL"/>
<node CREATED="1444110753735" ID="ID_1370602633" MODIFIED="1444110757278" TEXT="MemCached"/>
</node>
</node>
<node CREATED="1444035443687" ID="ID_1260737323" MODIFIED="1444036883991" TEXT="graph stores">
<node CREATED="1444040925612" ID="ID_1595135538" MODIFIED="1444076914693" TEXT="such as">
<node CREATED="1444040927558" ID="ID_1761730479" MODIFIED="1444040930622" TEXT="Neo4J"/>
<node CREATED="1444040933129" ID="ID_488105963" MODIFIED="1444040949626" TEXT="Pregel"/>
<node CREATED="1444076915485" ID="ID_876937415" MODIFIED="1444076918468" TEXT="Virtuoso"/>
<node CREATED="1444076918878" ID="ID_1889346688" MODIFIED="1444077023698" TEXT="IBM DB2 RDF Store"/>
<node CREATED="1444076933675" ID="ID_1466817522" MODIFIED="1444076950800" TEXT="Oracle RDF Semantic Graph"/>
<node CREATED="1444817748927" ID="ID_1116271271" MODIFIED="1475056306568" TEXT="CliqueSquare"/>
</node>
</node>
<node CREATED="1444036888602" ID="ID_906427212" MODIFIED="1475056477278" TEXT="MapReduce systems">
<node CREATED="1444036988184" ID="ID_2139514" MODIFIED="1444036990964" TEXT="extendedBy">
<node CREATED="1444036990966" ID="ID_568407588" MODIFIED="1475056282125" TEXT="Flink"/>
<node CREATED="1444036998695" ID="ID_1179998547" MODIFIED="1444037001747" TEXT="Spark"/>
<node CREATED="1444476967548" ID="ID_263174616" MODIFIED="1444476975010" TEXT="Tez from Apache"/>
</node>
<node CREATED="1444037166579" ID="ID_753792452" MODIFIED="1444037212065" TEXT="run as back-end for">
<node CREATED="1444037212066" ID="ID_184716877" MODIFIED="1444040477194" TEXT="Hive from Apache"/>
<node CREATED="1444037214419" ID="ID_738883221" MODIFIED="1444040480804" TEXT="Pig from Apache"/>
<node CREATED="1444038973544" ID="ID_1161071500" MODIFIED="1444040484107" TEXT="HBase from Apache"/>
</node>
<node CREATED="1444038769338" ID="ID_1639599755" MODIFIED="1444038789266" TEXT="based on">
<node CREATED="1444038779887" ID="ID_839612164" MODIFIED="1475056292233" TEXT="MapReduce model">
<node CREATED="1444038801487" ID="ID_984074261" MODIFIED="1444038802816" TEXT="extends">
<node CREATED="1444038802817" ID="ID_764660755" MODIFIED="1444040887701" TEXT="Bulk-synchronous parallelism (BSP) ">
<font NAME="SansSerif" SIZE="12"/>
</node>
</node>
</node>
</node>
<node CREATED="1444038905809" ID="ID_266293281" MODIFIED="1444038912851" TEXT="such as">
<node CREATED="1444038907369" ID="ID_1400514163" MODIFIED="1475056166430" TEXT="Hadoop from Apache"/>
<node CREATED="1444038912852" ID="ID_1436481104" MODIFIED="1444038915096" TEXT="Hortonworks"/>
<node CREATED="1444038916120" ID="ID_1848710747" MODIFIED="1444038924368" TEXT="MapR"/>
</node>
</node>
<node CREATED="1444731536808" ID="ID_1742807542" MODIFIED="1444731558897" TEXT="Streaming engine systems">
<node CREATED="1444731574928" ID="ID_830474969" MODIFIED="1444731592821" TEXT="such as">
<node CREATED="1444731598903" ID="ID_695725247" MODIFIED="1444731601764" TEXT="S4"/>
<node CREATED="1444731610505" ID="ID_1165818620" MODIFIED="1444731612955" TEXT="Storm"/>
<node CREATED="1444731621196" ID="ID_1771331352" MODIFIED="1444731626776" TEXT="Samza"/>
<node CREATED="1444731639481" ID="ID_1023472163" MODIFIED="1475056317994" TEXT="Flink"/>
</node>
</node>
<node CREATED="1444036893429" ID="ID_1405642597" MODIFIED="1475056454006" TEXT="P2P systems"/>
<node CREATED="1444037085101" ID="ID_505140590" MODIFIED="1475056464425" TEXT="mediator systems"/>
<node CREATED="1444037099152" ID="ID_1219885427" MODIFIED="1444037101701" TEXT="data warehouses"/>
<node CREATED="1444037239508" ID="ID_1494053050" MODIFIED="1475056384472" TEXT="&quot;Big Table&quot; systems">
<node CREATED="1444038885401" ID="ID_645469962" MODIFIED="1444038898299" TEXT="such as">
<node CREATED="1444038898301" ID="ID_1930779477" MODIFIED="1475056393569" TEXT="Cassandra from Apache"/>
<node CREATED="1444076474513" ID="ID_295540243" MODIFIED="1475056405656" TEXT="Spanner from Google"/>
<node CREATED="1444076509285" ID="ID_1523097290" MODIFIED="1475056413923" TEXT="F1 from Google"/>
<node CREATED="1444135521733" ID="ID_1343524175" MODIFIED="1475056422957" TEXT="Big Table"/>
</node>
</node>
<node CREATED="1444037255645" ID="ID_1970451864" MODIFIED="1444076712139">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p>
      BSP systems
    </p>
  </body>
</html></richcontent>
<node CREATED="1444037265227" ID="ID_765327530" MODIFIED="1444076716358" TEXT="such as">
<node CREATED="1444037267895" ID="ID_1272510758" MODIFIED="1444076725976" TEXT="Pregel from Google"/>
<node CREATED="1444076726357" ID="ID_1256923854" MODIFIED="1444076731324" TEXT="Giraph from Apache"/>
</node>
</node>
<node CREATED="1444076640035" ID="ID_389506026" MODIFIED="1444077051260" TEXT="Document stores">
<node CREATED="1444076643597" ID="ID_207490078" MODIFIED="1444076645305" TEXT="such as">
<node CREATED="1444076645306" ID="ID_50552708" MODIFIED="1475056431732" TEXT="MongoDB"/>
<node CREATED="1444076670153" ID="ID_385088687" MODIFIED="1444076679688" TEXT="CouchDB from Apache"/>
<node CREATED="1444076872915" ID="ID_1114980539" MODIFIED="1444076876584" TEXT="Oracle NoSQL"/>
<node CREATED="1444076956028" ID="ID_127313789" MODIFIED="1444076959633" TEXT="IBM PureXML"/>
<node CREATED="1444076960704" ID="ID_1136017416" MODIFIED="1444077019153" TEXT="IBM DB2 JSON"/>
</node>
</node>
<node CREATED="1444817678519" ID="ID_305078413" MODIFIED="1475056185289" TEXT="&quot;SQL over Hadoop&quot;">
<node CREATED="1444817681825" ID="ID_200111879" MODIFIED="1475056351005" TEXT="see BigTable systems; also join algorithms and optimization over MapReduce"/>
</node>
</node>
</node>
</map>

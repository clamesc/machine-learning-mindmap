<map version="freeplane 1.6.0">
<!--To view this file, download free mind mapping software Freeplane from http://freeplane.sourceforge.net -->
<node TEXT="Machine Learning" LOCALIZED_STYLE_REF="styles.topic" FOLDED="false" ID="ID_1723255651" CREATED="1283093380553" MODIFIED="1447084455284"><hook NAME="MapStyle" background="#fcfce9" zoom="1.614">
    <properties fit_to_viewport="false;" show_note_icons="true"/>

<map_styles>
<stylenode LOCALIZED_TEXT="styles.root_node" STYLE="oval" UNIFORM_SHAPE="true" VGAP_QUANTITY="24.0 pt">
<font SIZE="24"/>
<stylenode LOCALIZED_TEXT="styles.predefined" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="default" MAX_WIDTH="600.0 px" STYLE="as_parent">
<font NAME="SansSerif" SIZE="10" BOLD="false" ITALIC="false"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.details"/>
<stylenode LOCALIZED_TEXT="defaultstyle.attributes">
<font SIZE="9"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.note"/>
<stylenode LOCALIZED_TEXT="defaultstyle.floating">
<edge STYLE="hide_edge"/>
<cloud COLOR="#f0f0f0" SHAPE="ROUND_RECT"/>
</stylenode>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.user-defined" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="styles.topic" COLOR="#18898b" STYLE="fork">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.subtopic" COLOR="#cc3300" STYLE="fork">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.subsubtopic" COLOR="#669900">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.important">
<icon BUILTIN="yes"/>
</stylenode>
<stylenode TEXT="Stichpunkt" COLOR="#000000" STYLE="fork" MAX_WIDTH="300.0 px">
<font NAME="Liberation Sans" SIZE="8" BOLD="true"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;</i></font>
    </p>
  </body>
</html>
</richcontent>
</stylenode>
<stylenode TEXT="Beschreibung" COLOR="#666666" STYLE="fork" MAX_WIDTH="300.0 px">
<font NAME="Liberation Sans" SIZE="8" BOLD="false" ITALIC="true"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;</i></font>
    </p>
  </body>
</html>
</richcontent>
</stylenode>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.AutomaticLayout" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="AutomaticLayout.level.root" COLOR="#000000">
<font SIZE="18"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,1" COLOR="#0033ff">
<font SIZE="16"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,2" COLOR="#00b439">
<font SIZE="14"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,3" COLOR="#990000">
<font SIZE="12"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,4" COLOR="#111111">
<font SIZE="10"/>
</stylenode>
</stylenode>
</stylenode>
</map_styles>
</hook>
<node TEXT="Decision Trees" STYLE_REF="Stichpunkt" FOLDED="true" POSITION="right" ID="ID_1953448722" CREATED="1447084609780" MODIFIED="1447084617072">
<node TEXT="Inference" STYLE_REF="Stichpunkt" ID="ID_1393718795" CREATED="1447086424355" MODIFIED="1453650358323"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Count samples in regions</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\pi_{c,\mathcal R} := p(z=c|\mathcal R)=\frac{n_{c,\mathcal R}}{\sum_{k\in C}n_{k,\mathcal R}}=\frac 1 {\mathcal D_{\mathcal R}}\sum_{i\in \mathcal D_{mathcal R}} \mathbb I (z_i=c)\\&#xa;y = \text{argmax}_c\; p(z=c|x) = \text{argmax}_c\; p(z=c|\mathcal R) = \text{argmax}_c\; n_c" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Optimal decision tree" STYLE_REF="Stichpunkt" ID="ID_666629686" CREATED="1447086605238" MODIFIED="1459285637970"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Top-Down splitting node-by-node using a greedy heuristic on training data. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>Iterating over all possible trees too expensive.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Impurity measures" STYLE_REF="Stichpunkt" FOLDED="true" ID="ID_590056971" CREATED="1447087033251" MODIFIED="1447087038212">
<node TEXT="Misclassification rate" STYLE_REF="Beschreibung" ID="ID_401583826" CREATED="1447086717262" MODIFIED="1459274583539">
<hook EQUATION="\text{err}_{\mathcal D} = \frac 1 {|\mathcal D|}\sum_{i\in \mathcal D}\mathbb I(y_i \neq z_i)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Problems with misclassification rate as splitting heuristic" STYLE_REF="Beschreibung" ID="ID_1361519748" CREATED="1453651701966" MODIFIED="1459285963459"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Sometimes not improving, eventhough Entropy and Gini are.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Entropy" STYLE_REF="Beschreibung" ID="ID_1185978079" CREATED="1447086726240" MODIFIED="1459274579378">
<hook EQUATION="\mathbb H(\pi) = -\sum_{c=1}^C \pi_c \log \pi_c" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;Expected number of bits needed to encode a randomly draw value of X (under most efficient code)</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Example" STYLE_REF="Beschreibung" ID="ID_1117983423" CREATED="1453659021547" MODIFIED="1453659058501">
<hook EQUATION="H(S)= - p^+ \log_2 p^+ - p^- \log_2 p^-" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Gain" STYLE_REF="Stichpunkt" ID="ID_1318510238" CREATED="1453659395656" MODIFIED="1453659428209">
<hook EQUATION="Gain(S,A) = H(Y) - H(Y|A)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Gini index" STYLE_REF="Beschreibung" ID="ID_1239261943" CREATED="1447086730192" MODIFIED="1453650687787">
<hook EQUATION="\text{Gini}(\pi)=\sum_{c=1}^C \pi_c (1-\pi_c) = 1 - \sum_c \pi_c^2" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Impurity measures" STYLE_REF="Beschreibung" ID="ID_1217011190" CREATED="1459285783327" MODIFIED="1513257734293">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_588885938" CREATED="1459285712742" MODIFIED="1459285798527">
<hook URI="ml_3716411122979495392.png" SIZE="1.0" NAME="ExternalObject"/>
</node>
</node>
</node>
<node TEXT="In regression tasks:" STYLE_REF="Beschreibung" ID="ID_1461610981" CREATED="1513263928426" MODIFIED="1513263941431"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Variance</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Stopp criteria" STYLE_REF="Stichpunkt" FOLDED="true" ID="ID_1682983495" CREATED="1447086957348" MODIFIED="1447087027309">
<node TEXT="Distribution in branch is pure" STYLE_REF="Beschreibung" ID="ID_1642141728" CREATED="1447088954535" MODIFIED="1447088998684"/>
<node TEXT="Maximum depth reached" STYLE_REF="Beschreibung" ID="ID_564787857" CREATED="1447088965004" MODIFIED="1447088998451"/>
<node TEXT="Number of samples in each branch below certain threshold" STYLE_REF="Beschreibung" ID="ID_1001198336" CREATED="1447088969942" MODIFIED="1447088998210"/>
<node TEXT="benefit of splitting is below certain threshold" STYLE_REF="Beschreibung" ID="ID_621505327" CREATED="1447088985454" MODIFIED="1447088997476"/>
</node>
<node TEXT="Which attribute to choose?" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1902253850" CREATED="1513263035322" MODIFIED="1513263044446">
<node TEXT="A good heuristic prefers features that split the data so that each successor node is as pure as possible" STYLE_REF="Beschreibung" ID="ID_307206430" CREATED="1513263065880" MODIFIED="1513263103633"/>
<node TEXT="In other words, we want a measure that prefers attributes that have a high degree of &quot;order&quot;." STYLE_REF="Beschreibung" ID="ID_736323747" CREATED="1513263106872" MODIFIED="1513263165198"/>
<node TEXT="=&gt; Entropy" STYLE_REF="Beschreibung" ID="ID_1811555091" CREATED="1513263166855" MODIFIED="1513263217795"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;- Measure for (un-)orderdness </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>- Entropy is the amount of information that is contained</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Overfitting" STYLE_REF="Stichpunkt" ID="ID_1922059896" CREATED="1447089103421" MODIFIED="1459589096158"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Low training error, possibly 0 </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">Testing error is comparably high</font></i>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="Training set" STYLE_REF="Beschreibung" ID="ID_1864289071" CREATED="1447089192931" MODIFIED="1451945219300"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;build tree</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Validation set" STYLE_REF="Beschreibung" ID="ID_707381090" CREATED="1447089197956" MODIFIED="1459274594312"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;Prune tree at the node that yields the highest error reduction, until for all nodes t:</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\text{err}_{\mathcal D_V}(T) &lt; \text{err}_{\mathcal D_V}(T\backslash T_t)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Pruning" STYLE_REF="Stichpunkt" ID="ID_873562436" CREATED="1453651931690" MODIFIED="1459286142652"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Grow tree maximally and then prune it to avoid non-improving steps stopping it early. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>Pruning T w.r.t. t means deleting all descendant nodes of t (but not t itself).</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Test set" STYLE_REF="Beschreibung" ID="ID_634786731" CREATED="1447089200507" MODIFIED="1447089590148"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;evaluate final model</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Random Forests" STYLE_REF="Stichpunkt" ID="ID_781757435" CREATED="1447089465061" MODIFIED="1459589097711"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Ensemble of decision trees</font></i>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="Use bootstrap sample for building trees" STYLE_REF="Beschreibung" ID="ID_1927886959" CREATED="1447089674882" MODIFIED="1459286476461"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Randomly draw N (size of training set) samples from tree. By sampling with replacement we only use ca. 63.2% of D for each tree.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Top-down greedy heuristic with small variation" STYLE_REF="Beschreibung" ID="ID_931979632" CREATED="1447089705834" MODIFIED="1459589536071"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;At each node, instead of picking the best of all possible tests, randomly select a subset of d &lt; D features and consider only tests on these features.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Choosing d" STYLE_REF="Beschreibung" ID="ID_834471402" CREATED="1459286545834" MODIFIED="1459286569868"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Choosing a small d creates variance between the trees, but if we choose d too small, we will build random trees with poor split choices and little individual predictive power. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>Choosing d too large will create an army of very similar trees, such that there is hardly any advantage over a single tree. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>The answer is somewhere in between!</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Feature Importance" STYLE_REF="Beschreibung" ID="ID_1826617478" CREATED="1513262554705" MODIFIED="1513262613917">
<node TEXT="Mean Decrease Impurity" STYLE_REF="Stichpunkt" ID="ID_357894597" CREATED="1513262594027" MODIFIED="1513262692124"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;When training a tree, it can be computed how much each feature decreases the weighted impurity in a tree. For a forest, the impurity decrease from each feature can be averaged and the features are ranked to this measure.</i></font>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="&quot;We show that random forest variable importance measures are a sensible means for variable selection in many applications, but are not reliable in situations where potential predictor variables vary in their scale of measurement or their number of categories.&quot;" STYLE_REF="Beschreibung" ID="ID_865673183" CREATED="1513263365763" MODIFIED="1513263517438"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Bias in random forest variable importance measures: Illustrations, sources and a solution - Carolin Strobl, 2007</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="When one of correlated features is used, the importance of others might be significantly reduced since effectively the impurity is already removed. As a consequence, they will have a lower reported importance." STYLE_REF="Beschreibung" ID="ID_1127849623" CREATED="1513263630794" MODIFIED="1513263745930">
<node TEXT="The effect is somewhat reduced in random forests, but not completely." STYLE_REF="Beschreibung" ID="ID_1762720026" CREATED="1513263766962" MODIFIED="1513263793117"/>
</node>
</node>
<node TEXT="Mean decrease accuracy" STYLE_REF="Stichpunkt" ID="ID_457212104" CREATED="1513264110947" MODIFIED="1513264565999"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Directly measure the impact of each feature on accuracy of the model. The general idea is to permute the values of each feature and measure how much the permutation decreases the accuracy of the model. Clearly, for unimportant variables, the permutation should have little to no effect on model accuracy, while permuting important variables should significantly decrease it.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="kNN" STYLE_REF="Stichpunkt" FOLDED="true" POSITION="right" ID="ID_172347183" CREATED="1447238107478" MODIFIED="1447240467452">
<node TEXT="K-nearest-neighbor classification" STYLE_REF="Stichpunkt" ID="ID_1025826203" CREATED="1447238333485" MODIFIED="1459274835668"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;Vector will be labeled by the mode of its neighbors' labels:</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="y = \underset{c}{\operatorname{argmax}} \frac 1 K \sum_{i\in Nhood} Id(c = z_i)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
</node>
<node TEXT="K-nearest-neighbor regression" STYLE_REF="Stichpunkt" ID="ID_821804875" CREATED="1447238485434" MODIFIED="1459678743210"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;Vector will be labeled by a weighted mean of its neighbors' values.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="y = \frac 1 C \sum_{i\in Nhood} \frac 1 {d(x,x_i)}z_i\\&#xa;C = \sum_{i\in \mathcal N_K(x)} \frac 1 {d(x,x_i)}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Distance metrics" STYLE_REF="Stichpunkt" ID="ID_1204022921" CREATED="1447239856405" MODIFIED="1459274827125">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Euclidian distance" STYLE_REF="Beschreibung" ID="ID_1349593273" CREATED="1447239863388" MODIFIED="1453743020214" VSHIFT_QUANTITY="4.0 px">
<hook EQUATION="\sqrt{\sum_i (u_i-v_i)^2}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="L_1" STYLE_REF="Beschreibung" ID="ID_1616167633" CREATED="1447239871112" MODIFIED="1453743031838">
<hook EQUATION="\sum_i |u_i-v_i|" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="L_infty" STYLE_REF="Beschreibung" ID="ID_950336241" CREATED="1447239884366" MODIFIED="1453743037935">
<hook EQUATION="\max_i |u_i-v_i|" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Angle" STYLE_REF="Beschreibung" ID="ID_801228003" CREATED="1447239887969" MODIFIED="1453743045938">
<hook EQUATION="\frac{u^Tv}{\|u\|\|v\|}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Hamming distance, Edit distance, ..." STYLE_REF="Beschreibung" ID="ID_5706459" CREATED="1447239901448" MODIFIED="1453653668573"/>
<node TEXT="Mahalanobis distance" STYLE_REF="Beschreibung" ID="ID_57415224" CREATED="1447239890603" MODIFIED="1459274831260">
<hook EQUATION="\sqrt{(u-v)^T\Sigma^{-1}(u-v)}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Learning distance metric" STYLE_REF="Stichpunkt" ID="ID_480089803" CREATED="1447241450396" MODIFIED="1459678954206" HGAP_QUANTITY="159.0 px" VSHIFT_QUANTITY="41.0 px"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;Neighborhood Component Analysis (NCA) </i></font>
    </p>
    <p>
      <font size="1" color="#666666"><i>Learn from training set </i></font>
    </p>
    <p>
      <font size="1" color="#666666"><i>Choose A, such that Validation set is classified into its known category.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\begin{align}mahalanobis(x_1,x_2)&amp;=(x_1-x_2)^T Q (x_1-x_2)\\&#xa;&amp;= (Ax_1-Ax_2)^T (Ax_1 - Ax_2)&#xa;\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="FreeNode"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="How do we define good?" STYLE_REF="Beschreibung" ID="ID_103637065" CREATED="1459288473150" MODIFIED="1459288548624"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;A should be chosen in such a way that when we take one exemplar from the training set and try to classify it based on the remaining training set, it should be classified into its known category. And this should hold for all exemplars in the training set (Leave One Out Cross Validation&#8212;LOOCV). </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>But an infinitesimal change in A can affect the classification performance by a finite amount.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Dont assign classes but use probabilities instead" STYLE_REF="Beschreibung" ID="ID_1818441403" CREATED="1447243294202" MODIFIED="1447243503729">
<hook EQUATION="p_{ij} = \frac{\exp(-\|Ax_i - Ax_j\|^2)}{\sum_{k \neq i}\exp(-\|Ax_i - Ax_k\|^2)}, \;\; p_{ii} = 0" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;Each training point i selects another point j as its neighbor with some probability p_ij</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Compute the probability p_i that point i will be correctly classified" STYLE_REF="Beschreibung" ID="ID_803679644" CREATED="1447243462299" MODIFIED="1453654753269">
<hook EQUATION="p_i = \sum_{j\in C_i} p_{ij}, \;\; C_i = \{j|c_i = c_j\}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Maximize expected number of correctly classified points:" STYLE_REF="Beschreibung" ID="ID_757179873" CREATED="1447243622763" MODIFIED="1459274877162">
<hook EQUATION="f(A) = \sum_i \sum_{j\in C_i} p_{ij} = \sum_i p_i" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Gradient ascent" STYLE_REF="Beschreibung" ID="ID_1148045019" CREATED="1447243691089" MODIFIED="1453654508857">
<hook EQUATION="\frac{\partial f}{\partial A}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="By restricting A to be a non-square matrix of size d x D, NCA can also do linear dimensionality reduction" STYLE_REF="Beschreibung" ID="ID_1372342674" CREATED="1447243806961" MODIFIED="1447243841063"/>
</node>
</node>
</node>
</node>
<node TEXT="Parzen density estimator (Probabilistic interpretation)" STYLE_REF="Stichpunkt" ID="ID_392622566" CREATED="1447240768580" MODIFIED="1459590189585">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_480089803" STARTINCLINATION="70;-16;" ENDINCLINATION="-54;31;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Uniform weighted sum of Gaussian distributions centred on the training points. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>Neighborhood can be interpreted as standard deviation. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>Bayes' rule helps with a new datapoint x*:</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="p(c=b|x^*) \propto p(x^*|c=b)p(c=b)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Leave-one-out cross-validation (LOOCV)" STYLE_REF="Stichpunkt" ID="ID_1210510693" CREATED="1447239668704" MODIFIED="1459274842717">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_480089803" STARTINCLINATION="129;-38;" ENDINCLINATION="-50;71;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;Train on all but one sample. </i></font>
    </p>
    <p>
      <font size="1" color="#666666"><i>use as much data for training as possible, but still get good performance estimate. </i></font>
    </p>
    <p>
      <font size="1" color="#666666"><i>Need to train model N times</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="k-fold cross-validation" STYLE_REF="Stichpunkt" ID="ID_1359654743" CREATED="1447238617113" MODIFIED="1459274846496">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="train" STYLE_REF="Beschreibung" ID="ID_1397038492" CREATED="1447238630460" MODIFIED="1459678943191" VSHIFT_QUANTITY="6.0 px"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Split training set into k parts and use every part as validation set once. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>Average over all folds to get error estimate </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>Try different settings for hyper-parameters. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>Use all your training data and the best hyper-parameters for final&#160; training (and testing) of your model.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="test" STYLE_REF="Beschreibung" ID="ID_728311464" CREATED="1447238632747" MODIFIED="1447238635004"/>
</node>
</node>
<node TEXT="Standardization" STYLE_REF="Stichpunkt" ID="ID_381980228" CREATED="1447240012990" MODIFIED="1459287999151">
<hook NAME="AlwaysUnfoldedNode"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Scaling issues</i></font>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="Scale each feature to zero mean and unit variance" STYLE_REF="Beschreibung" ID="ID_1160322634" CREATED="1447240134214" MODIFIED="1447240168797">
<hook EQUATION="x_{std} = \frac {x-\mu}{\sigma}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Use Mahalanobis distance" STYLE_REF="Beschreibung" ID="ID_1073910348" CREATED="1447240188479" MODIFIED="1447240335754"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;-&gt; Diagonal Covariance matrix</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="The curse of dimensionality" STYLE_REF="Stichpunkt" ID="ID_447975818" CREATED="1447240604255" MODIFIED="1459288067538"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;N has to grow exponentially with the number of features.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="MLE/MAP/Bayes" STYLE_REF="Stichpunkt" FOLDED="true" POSITION="right" ID="ID_1899867888" CREATED="1447698114663" MODIFIED="1452606882437">
<node TEXT="MLE" STYLE_REF="Stichpunkt" ID="ID_1722923559" CREATED="1447698695978" MODIFIED="1459274721690"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Maximum </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">Likelihood </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">Estimate</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="MLE" STYLE_REF="Beschreibung" ID="ID_262160873" CREATED="1447700197748" MODIFIED="1453743876459">
<hook EQUATION="p(\mathcal D | \theta) = p(\mathcal D | \theta_1, \theta_2, ... , \theta_{n}) = \prod_{i=1}^{n} p_i(F_i=f_i|\theta_i) = \prod_{i=1}^{n}p(F_i = f_i | \theta)\\&#xa;\max_\theta p(\mathcal D | \theta) \Rightarrow \frac d {d\theta} \log \left(p(\mathcal D | \theta)\right) = 0\\" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;The logarithm is a monotonous function. </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">Sums are easier to derive than products.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="MAP" STYLE_REF="Stichpunkt" ID="ID_18232514" CREATED="1447698699912" MODIFIED="1459274726553"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Maximum </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">a posteriori </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">estimation</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Posterior probability" STYLE_REF="Stichpunkt" ID="ID_1797272365" CREATED="1447701444247" MODIFIED="1459448470389">
<hook EQUATION="\begin{align}p(\theta|\mathcal D) &amp;= \frac{p(\mathcal D | \theta)p(\theta)}{p(\mathcal D)}\\&#xa;&amp;= Beta(\ldots)\end{align}\\&#xa;\frac d {d\theta} p(\theta|\mathcal D) = 0" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Beta distribution</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Prior probability - Beta distribution" STYLE_REF="Stichpunkt" ID="ID_1321583288" CREATED="1447701711290" MODIFIED="1459448168269"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>Can be seen as the probability for a binary event. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>a&gt;0 and b&gt;0 denote the number of prior observations in a Bernoulli experiment.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="p(\theta | a,b) = \frac{\Gamma (a+b)}{\Gamma(a)\Gamma(b)} \theta^{a-1} (1-\theta)^{b-1}\\&#xa;\textrm{with }\Gamma(n) = (n-1)!\; \textrm{ if } n \in \mathbb N" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="helpful identity" STYLE_REF="Beschreibung" ID="ID_391474030" CREATED="1447707535593" MODIFIED="1459448284707">
<hook EQUATION="\int_0^1 Beta(\theta|a,b)d\theta=1\\&#xa;\int_0^1 \theta^{a-1}(1-\theta)^{b-1} d\theta = \frac {\Gamma(a)\Gamma(b)}{\Gamma(a+b)}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="How should a and b be chosen?" STYLE_REF="Beschreibung" ID="ID_394534775" CREATED="1447765936820" MODIFIED="1447765958364"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Simple Answer: Whatever works</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Probability of observed data" STYLE_REF="Stichpunkt" ID="ID_822410327" CREATED="1447707798725" MODIFIED="1447708283013">
<hook EQUATION="p(\mathcal D) = \int p(\mathcal D, \theta) d\theta = \int p(\mathcal D|\theta) p (\theta | a,b) d\theta" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
<node TEXT="Full Bayesian approach" STYLE_REF="Stichpunkt" ID="ID_1237960149" CREATED="1447765289987" MODIFIED="1447765900389">
<hook EQUATION="p(f|\mathcal D, a,b) = \int_0^1 p(f,\theta|\mathcal D, a,b)d\theta = \int_0^1 p(f|\theta)p(\theta|\mathcal D, a, b)d\theta" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Multivariate Gaussian" STYLE_REF="Stichpunkt" FOLDED="true" POSITION="right" ID="ID_38178486" CREATED="1447247463699" MODIFIED="1447247474612">
<node TEXT="Definition" STYLE_REF="Stichpunkt" ID="ID_1694508532" CREATED="1447247632390" MODIFIED="1447596173638">
<hook EQUATION="\mathcal{N} = \frac 1 {\sqrt{(2\pi)^d|\Sigma|}}\exp\left( -\frac 1 2 (x - \mu)^T \Sigma^{-1} (x-\mu) \right)" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;moment parameterisation</i></font>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="Normalization factor" STYLE_REF="Beschreibung" ID="ID_455151332" CREATED="1447595264003" MODIFIED="1459590977127">
<hook EQUATION="\int e^{-\frac{(x-\mu)^2}{2\sigma^2}}dx = \sqrt{2\pi\sigma^2}\\&#xa;f(x) = \exp\left( - \frac 1 2 (x-\mu)^T \Sigma^{-1} (x-\mu) \right)\\&#xa;\int f(x)dx = \sqrt{(2\pi)^d|\Sigma|}" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;1D and multivariate</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Interesting Observation" STYLE_REF="Beschreibung" ID="ID_1739699025" CREATED="1447595728969" MODIFIED="1447595873442"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;Multivariate distribution decomposes into product of univariate distributions. </i></font>
    </p>
    <p>
      <font size="1" color="#666666"><i>-&gt; For Gaussians, uncorrelated components induce independent components.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\textrm{diagonal } \Sigma \Rightarrow p(x) = \prod_i P(x_i)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Information form" STYLE_REF="Beschreibung" ID="ID_1382568985" CREATED="1447596178385" MODIFIED="1447596422155"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;natural (or: canonical) parameterisation</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\propto e^{-\frac 1 2 x^TAx+r^Tx}, \; A = \Sigma^{-1}, \; r = \Sigma^{-1}\mu" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Properties" STYLE_REF="Stichpunkt" FOLDED="true" ID="ID_240705872" CREATED="1447248406837" MODIFIED="1447248411057">
<node TEXT="In 2D, a bivariate Gaussian is depicted as an ellipse." STYLE_REF="Beschreibung" ID="ID_602556485" CREATED="1447247998972" MODIFIED="1447248026357"/>
<node TEXT="Sigma" STYLE_REF="Stichpunkt" ID="ID_1284945118" CREATED="1447247821178" MODIFIED="1459591182592">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="symmetric" STYLE_REF="Beschreibung" ID="ID_1475581726" CREATED="1447597292838" MODIFIED="1447597301092"/>
<node TEXT="positive definite" STYLE_REF="Beschreibung" ID="ID_1994331635" CREATED="1447597295512" MODIFIED="1447597300463"/>
</node>
<node TEXT="Closed under linear transformations" STYLE_REF="Stichpunkt" ID="ID_1712216137" CREATED="1447248417334" MODIFIED="1459591404148">
<hook EQUATION="Y = AX\\&#xa;E[Y] = A\mu\\&#xa;Cov[Y] = A\Sigma A^T\\&#xa;Y\sim \mathcal N (A\mu + \xi, \; A\Sigma A^T)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Central Limit Theorem" STYLE_REF="Stichpunkt" ID="ID_761993294" CREATED="1447249872776" MODIFIED="1459591187772"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Given certain conditions, the arithmetic mean of a sufficiently large number of iterates of independent random variables, each with a well-defined expected value and well-defined variance, will be approximately normally distributed, regardless of the unterlying distribution.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_285107605" CREATED="1459455147792" MODIFIED="1459455168042"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;if you average i.i.d. variables, then only mean and covariance are retained (everything else is smoothed away) and a Gaussian remains.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Maximum entropy distributions" STYLE_REF="Stichpunkt" ID="ID_85051952" CREATED="1447250448670" MODIFIED="1459591192476">
<hook EQUATION="X \sim \mathcal{N}(\mu, \Sigma) = \underset{P}{\operatorname{argmax}}(\mathcal{H}[P]\;|\;\mathbb{E}[X]=\mu, Cov[X]=\Sigma)" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Given mean &#956; and covariance &#931; (and nothing else!), what distribution has highest differential entropy?</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Upper bound on entropy" STYLE_REF="Beschreibung" ID="ID_1651866603" CREATED="1447250754073" MODIFIED="1447250783896">
<hook EQUATION="\frac 1 2 \log |2\pi e \Sigma|" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Cholesky decomposition" STYLE_REF="Stichpunkt" ID="ID_583764684" CREATED="1447593420555" MODIFIED="1459591196071">
<hook EQUATION="LL^T = \Sigma" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;L is lower triangluar with sitrctly positive diagonal entries.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Sampling" STYLE_REF="Beschreibung" ID="ID_480271608" CREATED="1447593635301" MODIFIED="1447593765081">
<hook EQUATION="Z\sim \mathcal{N} (0,I)\\&#xa;X = \mu + LZ \Rightarrow X \sim \mathcal{N}(\mu, \Sigma)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Product of two Gaussian pdfs" STYLE_REF="Stichpunkt" ID="ID_1687272263" CREATED="1447595963126" MODIFIED="1447596973252">
<hook EQUATION="\Sigma = (\Sigma_1^{-1}+\Sigma_2^{-1})^{-1}, \; \mu = \Sigma(\Sigma_1^{-1}\mu_1+\Sigma_2^{-1}\mu_2)" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;Multiply Gaussians in information form</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="MLE" STYLE_REF="Stichpunkt" ID="ID_738813100" CREATED="1447248780069" MODIFIED="1459591221820">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="mean" STYLE_REF="Beschreibung" ID="ID_1382752107" CREATED="1447248882448" MODIFIED="1459591201491">
<hook EQUATION="\mu" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Maximize negative loglikelihood" STYLE_REF="Beschreibung" ID="ID_948641409" CREATED="1447248980949" MODIFIED="1459591205326">
<hook EQUATION="X_i \approx \mathcal{N}(\mu,\Sigma), \; \mathcal{D} = \{x_i\}_{i=1}^{n}\\&#xa;\max_\mu\left[-\log\bigg(\prod_{i=1}^n \mathcal{N}(x_i|\mu, \Sigma)\bigg)\right]" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Optimum at algebraic mean" STYLE_REF="Beschreibung" ID="ID_1910748690" CREATED="1447249177539" MODIFIED="1447249223722">
<hook EQUATION="\mu_{MLE}^* = \frac 1 n \sum_{i=1}^nx_i" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
<node TEXT="covariance" STYLE_REF="Beschreibung" ID="ID_1829817321" CREATED="1447249315937" MODIFIED="1459591209907">
<hook EQUATION="\Sigma" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Maximize negative loglikelihood" STYLE_REF="Beschreibung" ID="ID_810108065" CREATED="1447249640755" MODIFIED="1459591214755">
<hook EQUATION="\max_{\Sigma^{-1}}\left[ l(\mu, \Sigma^{-1}) \right]" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Extremum at empirical covariance" STYLE_REF="Beschreibung" ID="ID_1774138901" CREATED="1447249726881" MODIFIED="1447249803373">
<hook EQUATION="\Sigma_{MLE}^* = \frac 1 n \sum_i (x_i-\mu)(x_i-\mu)^T" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
</node>
</node>
<node TEXT="Operations" STYLE_REF="Stichpunkt" ID="ID_1754130798" CREATED="1447597501723" MODIFIED="1513261030749">
<node TEXT="Marginalisation" STYLE_REF="Stichpunkt" ID="ID_1730333941" CREATED="1447597138130" MODIFIED="1459592585750">
<hook EQUATION="p(x_I)=?\\&#xa;X_I = I_I X\\&#xa;p(x_I)=\mathcal{N}(\mu_I,\Sigma_I)" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;Use Selection Matrix</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Conditioning" STYLE_REF="Stichpunkt" ID="ID_1956512762" CREATED="1447597496966" MODIFIED="1513261030751">
<hook EQUATION="p(x_I|x_R)=?\\&#xa;p(x)=p(x_I|x_R)p(x_R)\\&#xa;\mu_{I|R}=\mu_I + \Sigma_{IR}\Sigma_{RR}^{-1}(x_R - \mu_R)\\&#xa;\Sigma_{I|R} = \Sigma_{II} - \Sigma_{IR}\Sigma_{RR}^{-1}\Sigma_{RI}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Linear Gaussian Systems" STYLE_REF="Stichpunkt" ID="ID_785892490" CREATED="1447599823435" MODIFIED="1447599923501">
<hook EQUATION="p(x)=\mathcal{N}(\mu_X, \Sigma_X)\\&#xa;p(y|x) = \mathcal{N}(Ax+b, \Sigma_{Y|X})" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="p(x|y)" STYLE_REF="Beschreibung" ID="ID_353819000" CREATED="1447599961381" MODIFIED="1447600172217">
<hook EQUATION="p(x|y) = \mathcal{N}(x|\mu_{X|Y}, \Sigma_{X|Y})\\&#xa;\Sigma_{X|Y} = (\Sigma_X^{-1} + A^T\Sigma_{Y|X}^{-1}A)^{-1}\\&#xa;\mu_{Y|X} = \Sigma_{X|Y} \left( A^T\Sigma_{Y|X}^{-1}(y-b)+\Sigma_X^{-1}\mu_X \right)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="p(y)" STYLE_REF="Beschreibung" ID="ID_1623540379" CREATED="1447600192033" MODIFIED="1447600246912">
<hook EQUATION="p(y) = \mathcal{N}(y|A\mu_X + b, \Sigma_{Y|X}+A\Sigma_XA^T)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Bayes for Gaussian" STYLE_REF="Stichpunkt" ID="ID_1488691794" CREATED="1447600601754" MODIFIED="1459593038151"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;we want to determine the posterior distribution for &#956; from observations D = {x1 , x2 , . . . , xn }, where we assume that the covariance &#931; of these observations is known.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Gaussian prior" STYLE_REF="Beschreibung" ID="ID_683961601" CREATED="1447600648312" MODIFIED="1459592737413">
<hook EQUATION="p(\mu) = \mathcal{N}(\mu_0, V_0)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Posterior distribution" STYLE_REF="Beschreibung" ID="ID_1918479231" CREATED="1447600680020" MODIFIED="1453748589098">
<hook EQUATION="p(\mu|\mathcal{D}, \Sigma) = \mathcal{N}(\mu_n, V_n)\\&#xa;V_n = (V_0^{-1}+n\Sigma^{-1})^{-1}\\&#xa;\mu_n = V_n\left( \Sigma^{-1}\bigg(\sum_i x_i\bigg) + V_0^{-1}\mu_0 \right)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Uninformative prior" STYLE_REF="Beschreibung" ID="ID_1274673843" CREATED="1447600851162" MODIFIED="1459592740206">
<hook EQUATION="V_0^{-1} = 0I" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_1020328977" CREATED="1447600885789" MODIFIED="1447600944083">
<hook EQUATION="p(\mu|\mathcal{D}, \Sigma) = \mathcal{N}\left( \frac 1 n \sum_i x_i, \frac 1 n \Sigma \right)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
</node>
</node>
<node TEXT="Linear Regression" STYLE_REF="Stichpunkt" FOLDED="true" POSITION="right" ID="ID_148349516" CREATED="1447766188940" MODIFIED="1447766193828">
<node TEXT="Linear Model" STYLE_REF="Stichpunkt" ID="ID_427984399" CREATED="1447767193892" MODIFIED="1447781603020">
<hook EQUATION="y(x,w) = w_0 + \sum_{j=1}^{M-1}w_j\phi_j(x)=w^T\phi(x)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Offline / Batch Learning" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_221035172" CREATED="1447771457791" MODIFIED="1447771502018"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;All data points are available at once</font></i>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="Likelihood Function" STYLE_REF="Stichpunkt" ID="ID_1020832019" CREATED="1447767772550" MODIFIED="1459598400717">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1287311737" STARTINCLINATION="477;0;" ENDINCLINATION="477;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook EQUATION="z = y(x,w) + \epsilon \;\;\; [\epsilon: \textrm{ Gaussian, zero mean}]\\&#xa;z = (z_1, z_2, ..., z_N) \;\; X = (x_1, x_2,...,x_N)\\&#xa;\mathcal N (x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}\cdot e^{-\frac{(x-\mu)^2}{2\sigma^2}}\\&#xa;p(z|X,w,\sigma^2) = \prod^N \mathcal N (z_n|w^T\phi(x_n),\sigma^2)\\\ln p(z|X,w,\sigma^2) = \underbrace{\frac N 2 \ln \beta - \frac N 2 \ln(2\pi)}_{const.} - \beta \underbrace{\frac 1 2 \sum_{n=1}^N(z_n-w^T\phi(x_n))^2}_{\textrm{Minimize }E_{\mathcal D}(w)}\\&#xa;E_{\mathcal D}(w) = \frac 1 2 (z-\Phi w)^T (z - \Phi w)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Error function E_D</i></font>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="Design Matrix" STYLE_REF="Beschreibung" ID="ID_58447712" CREATED="1447769409819" MODIFIED="1447939074647" HGAP_QUANTITY="15.0 px" VSHIFT_QUANTITY="80.0 px">
<hook EQUATION="\Phi = \begin{pmatrix}&#xa;\phi_0(x_1) &amp; \cdots &amp; \phi_{M-1}(x_1)\\&#xa;\vdots &amp; \ddots &amp; \vdots\\&#xa;\phi_0(x_N) &amp; \cdots &amp; \phi_{M-1}(x_N)&#xa;\end{pmatrix}\\&#xa;\vec y = \Phi \vec w" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="MLE solution" STYLE_REF="Stichpunkt" ID="ID_536269709" CREATED="1447768820658" MODIFIED="1459597603625">
<hook EQUATION="\nabla_w \ln p(z|w,\beta) \propto \sum^N (z_n - w^T \phi(x_n))\phi(x_n)^T\\&#xa;\propto z \Phi^T - w^T\Phi^T\Phi = 0\\&#xa;\Rightarrow w_{ML} = \underbrace{(\Phi^T\Phi)^{-1}\Phi^T}_{\Phi^\dagger} z" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Moore-Penrose pseudo-inverse</i></font>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="Computational aspect" STYLE_REF="Beschreibung" ID="ID_1932957929" CREATED="1447770406533" MODIFIED="1459597680218">
<hook EQUATION="\textrm{- If $\Phi$ is not full rank, $(\Phi^T\Phi)^{-1}$ does not exist.}\\&#xa;\textrm{- Even if $\Phi$ is full rank, it can be ill-conditioned.}" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Computing the MLE solution w_ML using the normal equations is not such a great idea.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="SVD (Singular Value Decomposition)" STYLE_REF="Beschreibung" ID="ID_922147639" CREATED="1447770644715" MODIFIED="1447771195280">
<hook EQUATION="\Phi^T\Phi w = \Phi^T z\\&#xa;V\Sigma^T\Sigma V^T w = V \Sigma^TU^T z\\&#xa;\Rightarrow w = V\hat\Sigma U^T z\\&#xa;&#xa;\textrm{Out of all solutions that minimise $\|\Phi w-z\|_2^2$ \\this one has minimum $\|w\|_2^2$}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Regularisation" STYLE_REF="Stichpunkt" ID="ID_1574069200" CREATED="1447777824752" MODIFIED="1459593683060" HGAP_QUANTITY="28.0 px" VSHIFT_QUANTITY="-117.0 px"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;MLE often suffers from overfitting</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="E_{\mathcal D} = \frac 1 2 \sum^N(z_n-w^T\phi(x_n))^2 + \frac \lambda 2 \sum^M |w_j|^q" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="" ID="ID_381575722" CREATED="1447778505513" MODIFIED="1459593687969">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Bias" STYLE_REF="Stichpunkt" ID="ID_1937715485" CREATED="1447778179721" MODIFIED="1447778243736"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Difference between resulting model and generating model</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Variance" STYLE_REF="Stichpunkt" ID="ID_143386725" CREATED="1447778245028" MODIFIED="1447941102531">
<arrowlink SHAPE="CUBIC_CURVE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="8" FONT_FAMILY="SansSerif" DESTINATION="ID_1937715485" MIDDLE_LABEL="Tradeoff" STARTINCLINATION="46;0;" ENDINCLINATION="46;0;" STARTARROW="DEFAULT" ENDARROW="DEFAULT"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Variance in resulting models from different training sets</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Lagrange term" STYLE_REF="Beschreibung" ID="ID_958183199" CREATED="1447778569237" MODIFIED="1447778612292"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;This is like a Lagrange term specifying an additional constraint:</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\sum^M |w_j|^q \leq \eta" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Quadratic regulariser (l2, ridge regression): q=2, i.e." STYLE_REF="Beschreibung" ID="ID_434646114" CREATED="1447778614232" MODIFIED="1447942111979">
<hook EQUATION="\sum^M |w_j|^q = w^Tw\\&#xa;w_{ridge} = (\lambda I_M + \Phi^T\Phi)^{-1}\Phi^T z" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
<node TEXT="prior" STYLE_REF="Stichpunkt" ID="ID_561401552" CREATED="1447779297644" MODIFIED="1459598516501">
<hook EQUATION="p(w) = \mathcal N (w|m_0,S_0) \;\;[m_0: mean,\;S_0: covariance]" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;conjugate prior for a Gaussian with known variance is also a Gaussian.</i></font>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="posterior" STYLE_REF="Stichpunkt" ID="ID_1287311737" CREATED="1447779534808" MODIFIED="1459681341783" HGAP_QUANTITY="111.0 px" VSHIFT_QUANTITY="7.0 px">
<arrowlink SHAPE="CUBIC_CURVE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_434646114" STARTINCLINATION="7;-41;" ENDINCLINATION="-102;11;" STARTARROW="DEFAULT" ENDARROW="DEFAULT"/>
<hook EQUATION="p(w|z,\alpha,\beta) \propto p(z|w,\beta)p(w|\alpha)\\&#xa;\ln p(w|z,\alpha,\beta) = \frac \beta 2 \sum^N (z_n-w^T\phi(x_n))^2 + \frac \alpha 2 w^Tw + const.\\&#xa;p(w|z) = \mathcal N (w|m_N,S_N)\\&#xa;m_N = S_N (S_0^{-1}m_0 + \beta \Phi^T z)\\&#xa;S_N^{-1} = S_0^{-1} + \beta \Phi^T\Phi" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="MAP solution" STYLE_REF="Stichpunkt" ID="ID_1083140584" CREATED="1447780365663" MODIFIED="1459598527148">
<hook EQUATION="w_{MAP} = m_N" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
</node>
<node TEXT="Infinitely broad prior" STYLE_REF="Beschreibung" ID="ID_53258202" CREATED="1447780417880" MODIFIED="1447780489082">
<hook EQUATION="S_0^{-1} \rightarrow 0\\&#xa;w_{MAP} \rightarrow w_{ML} = \Phi^\dagger z" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="For N=0, i.e. no data points, we get the prior back" STYLE_REF="Beschreibung" ID="ID_886076434" CREATED="1447780494768" MODIFIED="1447780511478"/>
<node TEXT="simple example" STYLE_REF="Beschreibung" ID="ID_359044081" CREATED="1447781350546" MODIFIED="1459598844307">
<hook EQUATION="p(w|\alpha) = \mathcal N (w|m_0 = 0, S_0 = \alpha^{-1}I)\\&#xa;p(w|z) = \mathcal N (w|m_N,S_N)\\&#xa;m_N = \beta S_N \Phi^T z\\&#xa;S_N^{-1} = \alpha I + \beta \Phi^T\Phi" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
<node TEXT="Full Bayesian Predictive Distribution" STYLE_REF="Stichpunkt" ID="ID_1450523197" CREATED="1447781762816" MODIFIED="1459600386786">
<hook EQUATION="p(z|x,z,\alpha,\beta) = \int \underbrace{p(z|x,w,\beta)}_{likelihood}\underbrace{p(w|z,\alpha, \beta)}_{posterior}dw\\&#xa;p(z|x,z,\alpha,\beta) = \mathcal N (z|m_N^T\phi(x), \sigma_N^2(x))\\&#xa;\sigma_N^2(x) = \frac 1 \beta + \phi(x)^T S_N \phi(x)\\&#xa;\sigma_{N+1}^2(x) \leq \sigma_N^2(x)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Online Learning" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_287537489" CREATED="1447771490667" MODIFIED="1447771609369"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Batch processing of all points at once is infeasable. Data points arrive over time (sequentially) and possibly should be discarded as soon as possible.</font></i>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="LSM (Least-mean-squares)" STYLE_REF="Stichpunkt" ID="ID_660429871" CREATED="1447771792607" MODIFIED="1459682015471"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;Stochastic gradient descent </i></font>
    </p>
    <p>
      <font size="1" color="#666666"><i>update w after each newly arriving data point by applying the following technique</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="w^{(\tau+1)} = w^{(\tau)} - \eta \nabla E_n\\&#xa;w^{(\tau+1)} = w^{(\tau)} - \eta \frac \partial {\partial w^{(\tau)}} \log p(z^{(\tau)}|x^{(\tau)}, w^{(\tau)}, \beta)\\&#xa;w^{(\tau+1)} = w^{(\tau)} - \eta \left( z_n - (w^{(\tau)})^T \phi(x_n) \right) \phi(x_n)^T" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Error function of nth data point" STYLE_REF="Beschreibung" ID="ID_136269975" CREATED="1447772089582" MODIFIED="1447772113979">
<hook EQUATION="E_n" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Overall Error" STYLE_REF="Beschreibung" ID="ID_841538193" CREATED="1447772114617" MODIFIED="1447772131826">
<hook EQUATION="E = \sum_n E_n" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Learning rate needs to be chosen carefully to achieve convergence of the algorithm" STYLE_REF="Beschreibung" ID="ID_24364707" CREATED="1447772133606" MODIFIED="1447772169031">
<hook EQUATION="\eta" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Recursive Least Squares" STYLE_REF="Stichpunkt" ID="ID_743480101" CREATED="1447776254538" MODIFIED="1459682113407">
<hook EQUATION="R^{(\tau)} = (\Phi^{(\tau)})^T\Phi^{(\tau)}\\&#xa;w_{RLS}^{(\tau+1)} = w_{RLS}^{(\tau)} + (R^{(\tau+1)})^{-1}x^{(\tau+1)}\left( t^{(\tau+1)}-(x^{(\tau+1)})^T w_{RLS}^{(\tau)}\right)\\&#xa;(R^{(\tau+1)})^{-1} = (R^{\tau})^{-1} - (R^{(\tau)})^{-1}x^{(\tau+1)}\left( 1+(x^{\tau+1})^T (R^{(\tau)})^{-1}x^{(\tau+1)} \right)^{-1}(x^{(\tau+1)})^T(R^{(\tau)})^{-1}" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;Determine the optimal learning rate for linear regression. </i></font>
    </p>
    <p>
      <font size="1" color="#666666"><i>- Initial value for R_0^-1 is usually a diagonal matrix with large entries on its diagonal. </i></font>
    </p>
    <p>
      <font size="1" color="#666666"><i>&#160;- No matrix inversions are necessary </i></font>
    </p>
    <p>
      <font size="1" color="#666666"><i>&#160;- Versions with weighting/forgetting factors are also possible</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Linear Classification" STYLE_REF="Stichpunkt" POSITION="right" ID="ID_924268083" CREATED="1448822624942" MODIFIED="1448822630838">
<node TEXT="Classification Problem" STYLE_REF="Beschreibung" ID="ID_858303081" CREATED="1448822661161" MODIFIED="1448823586446"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Goal: Assign unknown input vector x to one of K classes.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\mathcal D = \{(x^n,z^n), n=1,\ldots,N\}\\&#xa;z \in \{0,1,\ldots,K-1\}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Disctriminative models" STYLE_REF="Beschreibung" ID="ID_1337893553" CREATED="1448829122705" MODIFIED="1459604062376">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1933798747" STARTINCLINATION="354;0;" ENDINCLINATION="354;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Hyperplanes" STYLE_REF="Beschreibung" ID="ID_691742228" CREATED="1448823020925" MODIFIED="1450364675483"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Let a plane be defined by its normal vector w and an offset b.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="x^Tw + b \begin{cases} =0 \;\text{if x on the plane} \\ &gt;0 \; \text{if x on normal&apos;s side of plane} \\ &lt;0 \; \text{else} \end{cases}" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="Advantage" STYLE_REF="Beschreibung" ID="ID_992457933" CREATED="1448823207577" MODIFIED="1453825525068"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Hyperplanes are computationally very convenient: easy to evaluate.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Linear Separability" STYLE_REF="Beschreibung" ID="ID_176792372" CREATED="1448823265947" MODIFIED="1453825515737"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;A data set D = {(x^n,z^n)} is linear separable if there exists a hyperplane for which all x^n with z^n=0 are on one and all x^n with z^n=1 on the other side.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Binary Classification" STYLE_REF="Beschreibung" ID="ID_434931605" CREATED="1448827193408" MODIFIED="1459604070052">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Zero-one loss" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_467649096" CREATED="1448827399525" MODIFIED="1448827409480">
<node TEXT="Zero-one loss" STYLE_REF="Beschreibung" ID="ID_1234834540" CREATED="1448823635736" MODIFIED="1459605725195">
<hook EQUATION="\sum_{i=1}^N \mathbb I (z^n = \hat z^n)" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;We are interested in the amount of samples we get right.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="The Perceptron" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1376076656" CREATED="1448827382623" MODIFIED="1448827387746">
<node TEXT="The Perceptron" STYLE_REF="Beschreibung" ID="ID_1699065842" CREATED="1448823693879" MODIFIED="1459605789495">
<hook EQUATION="\hat z = f(b + x^Tw)\\&#xa;f(\xi) = \begin{cases}1 \;if\; \xi &gt; 0 \\ 0 \;else\; \end{cases}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;A historical algorithm for binary classification.</i></font>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="Learining Rule" STYLE_REF="Beschreibung" ID="ID_936228699" CREATED="1448823945739" MODIFIED="1459604151420">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Initialize parameters to any value, e.g. a zero vector" STYLE_REF="Beschreibung" ID="ID_1215567662" CREATED="1448823970707" MODIFIED="1448823995731">
<hook EQUATION="w \leftarrow 0" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="While there is at least one misclassified x_i in the training set:" STYLE_REF="Beschreibung" ID="ID_1019040571" CREATED="1448823997497" MODIFIED="1453825737106">
<hook EQUATION="w \leftarrow \begin{cases} w + x^i, \;\text{ if}\; z^i = 1, \\ w - x^i, \;\text{ if}\; z^i = 0. \end{cases}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="What is the learning rule for the bias?" STYLE_REF="Beschreibung" ID="ID_1498132727" CREATED="1448824077222" MODIFIED="1453825769542">
<hook EQUATION="b \leftarrow \begin{cases} b+1, \;\text{ if}\; z^i = 1,\\b-1,\;\text{ if}\;z^i=0.\end{cases}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="This method converges to a w discriminating between two classes if it exists." STYLE_REF="Beschreibung" ID="ID_1034657588" CREATED="1448824135463" MODIFIED="1448824161438"/>
</node>
</node>
</node>
<node TEXT="Logistic Regression" STYLE_REF="Beschreibung" ID="ID_661768104" CREATED="1448827368053" MODIFIED="1448827373771">
<node TEXT="Logistic Regression" STYLE_REF="Beschreibung" ID="ID_170276379" CREATED="1448824929953" MODIFIED="1462299933759">
<hook EQUATION="p(z=1|x) = \sigma (b+x^Tw)\\&#xa;p(z=0|x) = 1 - \sigma (b+x^Tw)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;In discriminative models, we model the conditional of the output given the input, but not the input. That is, we model p(z | x ), but not p(x ). </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>Logistic regression is a discriminative model for classification: the output of our model is the parameter of a Bernoulli variable.</i></font>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="logit function" STYLE_REF="Beschreibung" ID="ID_839915220" CREATED="1448825062527" MODIFIED="1448825092630">
<hook EQUATION="\sigma(x) = \frac 1 {1+e^{-x}}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="MLE" STYLE_REF="Beschreibung" ID="ID_850451179" CREATED="1448825310407" MODIFIED="1459604165515">
<hook EQUATION="p(\{z^i\} | b,w,\{x^i\}) = \prod_{n=1}^N p(z^n|x^n,b,w)\\&#xa;= \prod_{n=1}^N \underbrace{p(z=1|x^n,b,w)^{z^n}}_{=1 \text{ if } z^n=0} \underbrace{(1 - p(z=1|x^n,b,w))^{1-z^n}}_{=1 \text{ if }\;z^n=1}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Log-likelihood" STYLE_REF="Beschreibung" ID="ID_779965099" CREATED="1448825873977" MODIFIED="1448826102570"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;This loss function is harder to optimize than the one for linear regression - there is no closed form available.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="L(w,b) = \log p(\{z^i\} | b,w,\{x^i\}) \\&#xa;= \sum_{n=1}^N z^n \log \sigma (b+w^Tx^n) + (1-z^n)\log(1-\sigma(b+w^Tx^n))" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Convex function -&gt; Gradient ascent" STYLE_REF="Beschreibung" ID="ID_1820159256" CREATED="1448826492041" MODIFIED="1448826623557">
<hook EQUATION="\nabla_w L = \sum_{n=1}^N (z^n - \sigma(b+w^Tx^n))x^n,\\&#xa;\frac{dL}{db} = \sum_{n=1}^N(z^n - \sigma(b+w^Tx^n))." NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
<node TEXT="Chi-square probability distribution" STYLE_REF="Beschreibung" ID="ID_559288055" CREATED="1462295166367" MODIFIED="1462295179828">
<node TEXT="Random Variable" STYLE_REF="Beschreibung" ID="ID_718232015" CREATED="1462295249417" MODIFIED="1462295517903">
<hook EQUATION="Q_1=X^2\\&#xa;Q_1\sim \mathcal\chi^2_1\\&#xa;Q_2=X_1^2 + X_2^2\\&#xa;Q_2\sim\mathcal\chi_2^2" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Examples</font></i>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="Normal distribution" STYLE_REF="Beschreibung" ID="ID_1273253755" CREATED="1462295184707" MODIFIED="1462295385889">
<hook EQUATION="X_i\sim\mathcal N(0,1)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
</node>
<node TEXT="Soft Zero-One Loss" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1600860495" CREATED="1448827356794" MODIFIED="1448827364985">
<node TEXT="Soft Zero-One Loss" STYLE_REF="Beschreibung" ID="ID_577752790" CREATED="1448826684070" MODIFIED="1448827344233">
<hook EQUATION="\sum_{n=1}^N [\sigma(\beta(b+w^Tx^n))-z^n]^2 + \lambda w^T w" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;- Becomes zero-one loss for beta -&gt; infty </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- Lambda is used to control the complexity of the model to prevent overfitting. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- The objective is no longer convex: that means that we are no longer guaranteed to find the optimum.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Hinge loss" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1150728520" CREATED="1448827346657" MODIFIED="1448827351411">
<node TEXT="Hinge loss" STYLE_REF="Beschreibung" ID="ID_1765272854" CREATED="1448826987429" MODIFIED="1448827343180">
<hook EQUATION="\sum_{n=1}^N \max(0,1-y\tilde z^n),\\&#xa;\text{where }\tilde z^n = 2z^n-1" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;- Although it is only locally differentiable, it works very well. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- A variant, quared hinge loss is a reasonable alternative. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- Objective is convex.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Multiclass problems" STYLE_REF="Beschreibung" ID="ID_243928008" CREATED="1448828680175" MODIFIED="1459604187099">
<hook EQUATION="z = (0,0,1,0,0,0)^T" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS" HIDDEN="true">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;We then also have multiple w, stacked into the columns of a matrix W. The bias is a vector as well, b.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Logistic Regression" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1970812883" CREATED="1448828963729" MODIFIED="1448828968777">
<node TEXT="Softmax function" STYLE_REF="Beschreibung" ID="ID_151315018" CREATED="1448828777621" MODIFIED="1459606791674">
<hook EQUATION="p(z=i|x) = \frac{\exp(b_i + w_i^Tx)}{\sum_j\exp(b_j+w_j^Tx)}" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;If we have multiple classes, we use multiple logistic regression models and normalize so the outputs sum up to 1. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>For learning, we need to write down the log-likelihood and its derivatives again. </i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Multiclass hinge loss" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1731810768" CREATED="1448828970972" MODIFIED="1448828976940">
<node TEXT="Multiclass hinge loss" STYLE_REF="Beschreibung" ID="ID_371149696" CREATED="1448828977798" MODIFIED="1459606807640">
<hook EQUATION="y = {\operatorname{argmax}}_j x^Tw_j + b_j" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Train each w j in a one-vs-all fashion.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
</node>
<node TEXT="Generative models" STYLE_REF="Beschreibung" ID="ID_1457436806" CREATED="1448829205094" MODIFIED="1513444871618"><richcontent TYPE="DETAILS" HIDDEN="true">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Intuitive idea: For each class, estimate a model. For a new input x, check to which model it fits best. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">Formal idea: Use p(z=k) and p(x|z=k) and Bayes to get p(z=k|x)</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Outliers" STYLE_REF="Beschreibung" ID="ID_1933798747" CREATED="1448830708063" MODIFIED="1448830977710" HGAP_QUANTITY="26.0 px" VSHIFT_QUANTITY="-2.0 px"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Logistic regression and the generative model we showed are not robust towards outliers. The chance of a single point to be far from right side of the decision boundary is exponentially small.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="K&gt;2 classes" STYLE_REF="Beschreibung" ID="ID_66210560" CREATED="1448829735813" MODIFIED="1459604083299">
<hook EQUATION="p(z=k|x) = \frac{p(x|z=k)p(z=k)}{\sum_j p(x|z=j)p(z=j)}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Model for the class priors" STYLE_REF="Beschreibung" ID="ID_650180008" CREATED="1448829876383" MODIFIED="1448830141496"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;The maximum likelihood estimator for this is the fraction of samples that fall into the class - plain counting.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="0 \leq \Theta_i \leq 1,\;\; \sum_i \Theta_i = 1\\&#xa;p(z=i|\Theta) = \Theta_i" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Class conditionals" STYLE_REF="Beschreibung" ID="ID_1692156833" CREATED="1448830241772" MODIFIED="1459604086798"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Assume Multivariate Gaussian</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="p(x|z=k) = \frac 1 {|2\pi\Sigma|^{1/2}} \exp \left( - \frac 1 2 (x- \mu_k)^T \Sigma^{-1} (x - \mu_k) \right)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Estimate Parameters" STYLE_REF="Beschreibung" ID="ID_963786942" CREATED="1448830410417" MODIFIED="1459607575783"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;We can estimate all the parameters by MLE for the MVG on the subset corresponding to the class of the data set. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>For covariance, we take the full data set.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="K=2 classes" STYLE_REF="Beschreibung" ID="ID_1974496519" CREATED="1448829727297" MODIFIED="1459607623426">
<hook EQUATION="p(z=1|x) = \frac{p(x|z=1)p(z=1)}{p(x|z=0)p(z=0) + p(x|z=1)p(z=1)}\\&#xa;p(z=1|x) = \frac 1 {1 + \exp(-(w^Tx+b))} = \sigma(w^Tx+b)\\&#xa;w = \Sigma^{-1}(\mu_0-\mu_1)\\&#xa;b = - \frac 1 2 \mu_0^T\Sigma^{-1}\mu_0 + \frac 1 2 \mu_1^T\Sigma^{-1}\mu_1 + \ln \frac {p(z=1)}{p(z=0)}" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;It&#8217;s the same model as with logistic regression. But the parameters are estimated differently.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Kernels" STYLE_REF="Stichpunkt" FOLDED="true" POSITION="right" ID="ID_1543304828" CREATED="1450365108875" MODIFIED="1450365832432">
<node TEXT="Dual representation" STYLE_REF="Stichpunkt" FOLDED="true" ID="ID_320475141" CREATED="1450368269805" MODIFIED="1459637069491">
<hook EQUATION="w = \sum_{n=1}^N a_n x^{(n)}\\&#xa;\Rightarrow w = \sum_{n=1}^N a_n \phi(x^{(n)})" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="Null space of training set" STYLE_REF="Beschreibung" ID="ID_635435974" CREATED="1450367972214" MODIFIED="1459613607198">
<hook EQUATION="\mathcal X = \{ x^{(n)}, n=1,\ldots,N \}\\&#xa;y(x) = w^Tx,\;\;x\in\mathcal X\\&#xa;w = \hat w + z,\;\;&#xa;\hat w \in  span(\mathcal X),\;z\in \mathcal X^\perp\\&#xa;y(x) = (\hat w + z)^T x = \hat w^T x + z^T x = \hat w^T x\\&#xa;\Rightarrow w \in span(\mathcal X)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Orthogonal Complement" STYLE_REF="Beschreibung" ID="ID_611573818" CREATED="1450367057721" MODIFIED="1450370884797">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1256551691" STARTINCLINATION="331;0;" ENDINCLINATION="331;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook EQUATION="\mathcal A^\perp = \{ z\in \mathbb R^N : x^T z = 0\; \forall x \in \mathcal A \}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Orthogonal Decomposition" STYLE_REF="Beschreibung" ID="ID_1950374195" CREATED="1450367300939" MODIFIED="1459613593077"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Any y in R^N can be written uniquely in the form</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="y = \hat y + z\\&#xa;(\hat y \in \mathcal A, \; z \in \mathcal A^\perp)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Example" STYLE_REF="Beschreibung" ID="ID_1256551691" CREATED="1450367449346" MODIFIED="1459613596561" HGAP_QUANTITY="19.0 px" VSHIFT_QUANTITY="-26.0 px">
<hook EQUATION="\mathcal A = \{ c_1(1,0,0) + c_2(1,1,0), c\in \mathbb R^2 \}\\&#xa;\mathcal A^\perp = \{ c(0,0,1), c \in \mathbb R \}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_430547241" CREATED="1450367458452" MODIFIED="1450367541749">
<hook EQUATION="y = (2,3,5)\\&#xa;\hat y = (2,3,0)\\&#xa;z = (0,0,5)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
<node TEXT="Kernel Function" STYLE_REF="Stichpunkt" ID="ID_1090275224" CREATED="1450368901556" MODIFIED="1459613727879">
<hook EQUATION="y(x) = w^T \phi(x) = \sum_{n=1}^N a_n(\phi^{(n)})^T\phi(x) = \sum_{n=1}^N a_n K(x^{(n)},x)\\&#xa;\Rightarrow K(x,y) := \phi(x)^T\phi(y)" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;The weight vector w is thus a linear combination of the training samples. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>There are as many dual parameters as training samples. Their number is independent of the number of basis functions. </i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Gram matrix" STYLE_REF="Beschreibung" ID="ID_453722415" CREATED="1450369143358" MODIFIED="1450369286356">
<hook EQUATION="K_{mn} := K(x^{(m)},x^{(n)}) = \phi(x^{(m)})^T\phi(x^{(n)})\\&#xa;\text{the predictions } Y_n = y(x^{(n)}) \text{ on the training set are simply}\\&#xa;Y = a^T K" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Gaussian kernel" STYLE_REF="Beschreibung" ID="ID_1909745304" CREATED="1450372141803" MODIFIED="1459604369834">
<hook EQUATION="y(x) = \sum_{n=1}^N a_n K(x^{(n)},x)\\&#xa;K(x,y) = \exp\left( - \frac{|x-y|^2}{2\sigma^2} \right)\\&#xa;y(x) = \sum_{n=1}^N a_n \exp\left( - \frac{|x - x^{(n)}|^2}{2\sigma^2} \right)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Variance" STYLE_REF="Beschreibung" ID="ID_1427537651" CREATED="1450375022339" MODIFIED="1450375062472"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;The quality of the result is very sensitive to the choice of the variance. Use cross-validation to choose the right value.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Radial basis functions (RBFs)" STYLE_REF="Stichpunkt" ID="ID_1037996886" CREATED="1450365951090" MODIFIED="1459762391374"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Approximate the function with a sum of bump-shaped functions: </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>Gaussian Radial Basis Functions:</i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>The center is at m(i) and alpha determines the width of the bump.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\phi_i(x) = \exp\left( - \frac{(x-m^{(i)})^2}{2\alpha^2} \right)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Advantage" STYLE_REF="Beschreibung" ID="ID_1014654140" CREATED="1450366108304" MODIFIED="1450366148260"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;The decrease smoothly to zero and do not show oscillatory behavior unlike higher order polynomials.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Disadvantage" STYLE_REF="Beschreibung" ID="ID_1571036424" CREATED="1450366303808" MODIFIED="1459612986458"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;To cover the data with a constant discretization level (number of RBFs per unit volume) the number of basis functions and weights grows exponentially with the number of dimensions.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Dual function" STYLE_REF="Beschreibung" ID="ID_855277267" CREATED="1459612911533" MODIFIED="1459612973871"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Reduce number of required weights</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
</node>
<node TEXT="Kernel of polynomial basis" STYLE_REF="Beschreibung" ID="ID_752196752" CREATED="1450370477060" MODIFIED="1459613977684"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Evaluating the kernel is much cheaper than calculating the mapping phi into feature space and the scalar product explicitly.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\phi_i(x) = x^{i-1}, \;\; i=1,\ldots,D\\&#xa;K(x,y) = \phi(x)^T \phi(y) = \sum_{i=1}^D x^{i-1}y^{i-1} = \sum_{i=0}^{D-1} (xy)^i = \frac{1 - (xy)^D}{1-xy}\\&#xa;K_{nm} = \frac {1 - (x^{(n)}x^{(m)})^D}{1 - x^{(n)}x^{(m)}}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
<node TEXT="Linear Regression" STYLE_REF="Beschreibung" ID="ID_439193190" CREATED="1450369950228" MODIFIED="1459613620841">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Primal representation" STYLE_REF="Beschreibung" ID="ID_1150360704" CREATED="1450369761514" MODIFIED="1459613627986">
<hook EQUATION="E(w) = |y - w^T\Phi|_2^2\\&#xa;w = (\Phi\Phi^T)^{-1} y = \Phi^+ y\\&#xa;y(x) = w^T \phi(x)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Complexity" STYLE_REF="Beschreibung" ID="ID_1963237492" CREATED="1450369874055" MODIFIED="1450370253443">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1411782631" STARTINCLINATION="349;0;" ENDINCLINATION="349;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Complexity of w is O(M^3) where M is the number of basis functions.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Dual representation" STYLE_REF="Beschreibung" ID="ID_1123638523" CREATED="1450369463475" MODIFIED="1459613624839">
<hook EQUATION="E(a) = |y-a^TK|_2^2\\&#xa;a = (KK^T)^{-1}K y = K^+ y\\&#xa;y(x) = \sum_{n=1}^N a_n K(x^{(n)},x)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Complexity" STYLE_REF="Beschreibung" ID="ID_1603250954" CREATED="1450369906428" MODIFIED="1459613632854">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1411782631" STARTINCLINATION="320;0;" ENDINCLINATION="320;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Complexity of a is O(N^3) where N is the number of training samples.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Saving parameters" STYLE_REF="Stichpunkt" ID="ID_1411782631" CREATED="1450370032126" MODIFIED="1450370263132" VSHIFT_QUANTITY="-14.0 px"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Lots of basis functions &amp; moderate number of training samples: </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>=&gt; dual representation saves lots of parameters</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
</node>
</node>
<node TEXT="Properties" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_94028619" CREATED="1450371008071" MODIFIED="1450371043987">
<node TEXT="Mercer&apos;s theorem (for finite input spaces)" STYLE_REF="Stichpunkt" ID="ID_1158263409" CREATED="1450371112894" MODIFIED="1459636589242"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Consider a finite input space X={x1,...,xN) with K(xn, xm) a function on X. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>Then K is a kernel function, that is a scalar product in a feature space, if and only if K is symmetric and the matrix K_nm is positive semi-definite.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Making kernels from kernels" STYLE_REF="Stichpunkt" ID="ID_1043469394" CREATED="1450371457340" MODIFIED="1459694271305">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_635222697" CREATED="1450371468820" MODIFIED="1450371958110">
<hook EQUATION="K(x,y) = K_1(x,y) + K_2(x,y)\\" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_1261605115" CREATED="1450371503450" MODIFIED="1459693991200">
<hook EQUATION="K(x,y)=a K_1(x,y)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_565620098" CREATED="1450371787342" MODIFIED="1450371802200"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;for a &gt; 0</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_1263857047" CREATED="1450371534853" MODIFIED="1450371964109">
<hook EQUATION="K(x,y) = K_1(x,y)K_2(x,y)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_998466582" CREATED="1450371571634" MODIFIED="1459693993627">
<hook EQUATION="K(x,y) = K_3(\phi(x),\phi(y))" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_282568424" CREATED="1450371730163" MODIFIED="1450371772861">
<hook EQUATION="\mathbb R^m\;and\;\phi:\mathcal X \rightarrow \mathbb R^m" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;for K3 kernel on</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_801858951" CREATED="1450371655725" MODIFIED="1459693996601">
<hook EQUATION="K(x,y) = x^T B y" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_257022875" CREATED="1450371676879" MODIFIED="1450371776922"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;for B symmetric and positive semi-definite n x n matrix</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Combining model-based and model-free approaches" STYLE_REF="Beschreibung" ID="ID_1902606593" CREATED="1450375212243" MODIFIED="1513523561439"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;If there is a feature space that partly explains the data, we can add the kernel induced by that feature space to a generic kernel K2 to obtain a better fit. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>Here, we could combine a linear model phi with the Gaussian Kernel K2. The linear model captures the general trend of data. The Gaussian kernel captures local variations.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="K(x,y) = \phi(x)^T \phi(y) + K_2(x,y)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Commonly used kernels" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_976091706" CREATED="1450375393130" MODIFIED="1450375399582">
<node TEXT="Linear" STYLE_REF="Beschreibung" ID="ID_1593155689" CREATED="1450375401526" MODIFIED="1450375419664">
<hook EQUATION="K(x,y) = x^Ty" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Polynomial" STYLE_REF="Beschreibung" ID="ID_1029350547" CREATED="1450375420340" MODIFIED="1450375635834">
<hook EQUATION="K(x,y) = ((x^Ty)+c)^d" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Gaussian" STYLE_REF="Beschreibung" ID="ID_197894835" CREATED="1450375638715" MODIFIED="1450375688685">
<hook EQUATION="K(x,y) = \exp\left(-\frac{|x-y|^2}{2\sigma^2}\right)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Not PSD" STYLE_REF="Beschreibung" ID="ID_438041630" CREATED="1450375712752" MODIFIED="1450375762811"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Sometimes functions are used as kernels that are not positive semi-definite. Usually this works but it can lead to strange results.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Constrained Convex Optimization" STYLE_REF="Stichpunkt" FOLDED="true" POSITION="right" ID="ID_149401749" CREATED="1450458457205" MODIFIED="1450458466534">
<node TEXT="Convexity" STYLE_REF="Beschreibung" ID="ID_1854245237" CREATED="1450459986224" MODIFIED="1450459989279">
<node TEXT="Convex function" STYLE_REF="Stichpunkt" ID="ID_1117957565" CREATED="1450458529265" MODIFIED="1459694297413"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;If a function is convex, any two points of the function's graph can be connected by a straight line that is above the function graph.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="f(tx + (1-t)y)\leq tf(x)+(1-t)f(y)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Concave function" STYLE_REF="Beschreibung" ID="ID_1303037889" CREATED="1450458849697" MODIFIED="1450458871497"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;A function is concave if and only if -f is convex</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="First-order convexity condition" STYLE_REF="Beschreibung" ID="ID_174481172" CREATED="1450458949323" MODIFIED="1459694288634"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Difference between f(y) and f(x) is bounded by function values between x and y.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="f(y) - f(x) \geq \frac{f(x+t(y-x))-f(x)}{t}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Theorem" STYLE_REF="Beschreibung" ID="ID_1224072399" CREATED="1450459092625" MODIFIED="1450459225384"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Suppose f : X -&gt; R is a differentiable function and X is convex. Then f is convex if and only if for x,y in X</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="f(y) \geq f(x)+ (y-x)^T \nabla f(x)." NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Minimization of convex functions" STYLE_REF="Beschreibung" ID="ID_815465187" CREATED="1450459372970" MODIFIED="1450459456581"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;At a (local) minimum point x* the gradient must be zero, otherwise we could follow the gradient to get an even lower value. If f is a convex function, then x* is a global minimum of f.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Convex set" STYLE_REF="Stichpunkt" ID="ID_350357041" CREATED="1450459516769" MODIFIED="1450459564522"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;A set X in a vector space is called convex if, for all x,y in X and any t in [0,1]</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="[(1-t)x+ty]\in\mathcal X" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Constrained optimization problem" STYLE_REF="Stichpunkt" ID="ID_1626318515" CREATED="1450459672623" MODIFIED="1450459821803">
<hook EQUATION="\begin{align}&amp;minimize &amp;&amp;f_0(x)\\&#xa;&amp;subject\;to\;&amp;&amp;f_i(x)\leq 0, \;i=1,\ldots,p&#xa;\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="Feasibility" STYLE_REF="Beschreibung" ID="ID_462555554" CREATED="1450459829845" MODIFIED="1450459880719"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;A point x in R^N is called feasible if and only if it satisfies all constraints of the optimization problem.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Minimum and minimizer" STYLE_REF="Beschreibung" ID="ID_1138049059" CREATED="1450459882284" MODIFIED="1450459951508"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;We call the optimal value the minimum p*, and the point where the minimum is obtained the minimizer x*. Thus p* = f(x*).</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Minimization with inequality constraints" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_897962361" CREATED="1450460973421" MODIFIED="1459695966979"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;For multiple constraints f_i we have reached the minimum when the negative gradient has no component that we could follow without changing the value of any constraint. </i></font>
    </p>
    <p>
      <font size="1" color="#666666"><i>This is the case when the gradient of f0 is a linear combination of the constraint gradients.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="-\nabla f_0(x*) = \sum_{i=1}^p \alpha_i \nabla f_i(x*), \; \alpha_i\geq 0" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="Lagrangian" STYLE_REF="Beschreibung" ID="ID_1335566724" CREATED="1450461365651" MODIFIED="1450461837793"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;We refer to alpha_i as the Lagrange multiplier associated with the inequality constraint f_i. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>Calculating the gradient of L w.r.t. x shows that our optimality criterion at x* is recovered.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="L(x,\alpha) = f_0(x) + \sum_{i=1}^m \alpha_i f_i(x)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Lagrange dual function" STYLE_REF="Beschreibung" ID="ID_734110196" CREATED="1450461886565" MODIFIED="1459694328198"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Minimum of the Lagrangian over x given alpha. It is concave since it is the pointwise minimum of a family of affine functions of alpha.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="g(\alpha) = \min_x L(x,\alpha)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Lower bound" STYLE_REF="Beschreibung" ID="ID_1971945591" CREATED="1450462086308" MODIFIED="1450462156823"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;For any alpha &gt;= 0 the value of the Lagrange dual function g is a lower bound of the minimum p* of the constrained problem,</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="g(\alpha)\leq p*" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Lagrange dual problem" STYLE_REF="Beschreibung" ID="ID_489344589" CREATED="1450462602790" MODIFIED="1459694332182">
<hook EQUATION="\begin{align} &amp;maximize &amp;&amp; g(\alpha) = \min_x L(x,\alpha)\\&#xa;&amp;subject\;to &amp;&amp; \alpha_i\geq 0,\;i=1,\ldots,m\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;The maximum d* of the Lagrangian dual problem is the best lower bound on p* that we can achieve by using the Lagrangian.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Weak duality" STYLE_REF="Beschreibung" ID="ID_71794185" CREATED="1450462867398" MODIFIED="1450462895585">
<hook EQUATION="d^* \leq p^*" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Strong duality" STYLE_REF="Beschreibung" ID="ID_1500833663" CREATED="1450462904099" MODIFIED="1459694334365">
<hook EQUATION="d^* = p^*" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Slaters constraint qualification" STYLE_REF="Beschreibung" ID="ID_1918486331" CREATED="1450463162995" MODIFIED="1459694337026">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_1903659234" CREATED="1450463004249" MODIFIED="1450463116414"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;If f0, f1, ... ,fm are convex and there exists an x in R^n such that</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="f_i(x)&lt;0,\;i=1,\ldots,m" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_117856443" CREATED="1450463118427" MODIFIED="1450464298748"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;or the constraints are affine, that is</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="f_i(x) = w_i^T x + b_i \leq 0" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_137732886" CREATED="1459696678127" MODIFIED="1459696695200"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;the duality gap is zero.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Dual solution" STYLE_REF="Beschreibung" ID="ID_1970608158" CREATED="1450463439179" MODIFIED="1459694338882"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Let x* be a minimizer of the primal problem and alpha* a maximizer of the dual problem. If strong duality holds, then</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="L(x^*,\alpha^*) = g(\alpha^*) = \min_x L(x,\alpha^*)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Corollary" STYLE_REF="Beschreibung" ID="ID_1476357490" CREATED="1450463743289" MODIFIED="1450463865012"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Let x* be a minimizer of the primal problem and alpha* a maximizer of the dual problem. If strong duality holds and L is convex in x, then</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="x^* = \underset{x}{\operatorname{argmin}} L(x,\alpha^*)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
</node>
</node>
<node TEXT="Recipe" STYLE_REF="Stichpunkt" FOLDED="true" ID="ID_907137108" CREATED="1450464382848" MODIFIED="1450464443290">
<node TEXT="1. Calculate the Lagrangian" STYLE_REF="Beschreibung" ID="ID_26801970" CREATED="1450464430528" MODIFIED="1450464493679">
<hook EQUATION="L(x,\alpha) = f_0(x) + \sum_{i=1}^m \alpha_i f_i(x)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="2. Obtain the Lagrange dual function" STYLE_REF="Beschreibung" ID="ID_798162785" CREATED="1450464478461" MODIFIED="1450464536086">
<hook EQUATION="\nabla_{x^*} L(x^*,\alpha)=0" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="3. Solve the dual problem" STYLE_REF="Beschreibung" ID="ID_1703956" CREATED="1450464503501" MODIFIED="1450464616883">
<hook EQUATION="\begin{align}&amp;maximize &amp;&amp;g(\alpha)=L(x^*,\alpha)\\&#xa;&amp;subject\;to &amp;&amp;\alpha_i \geq 0,\;i=1,\ldots,m.\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="KKT Conditions" STYLE_REF="Beschreibung" ID="ID_1221210844" CREATED="1450464959498" MODIFIED="1459694316364"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;If and only if x*, a* are points that satisfy the Karush-Kuhn-Tucker conditions, for i=1,...,m, then x* and a* are optimal solutions of the constrained optimization problem and the corresponding Lagrange dual problem.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\begin{align} &amp;f_i(x^*)\leq 0 &amp;&amp;\text{primal feasibility},\\&#xa;&amp;\alpha_i^* \geq 0 &amp;&amp;\text{dual feasibility},\\&#xa;&amp;\alpha_i^* f_i(x^*)=0 &amp;&amp;\text{complementary slackness},\\&#xa;&amp;\nabla_x L(x^*,\alpha^*)=0 &amp;&amp;x^*\text{ minimizes Lagrangian}\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Complementary slackness" STYLE_REF="Beschreibung" ID="ID_224961036" CREATED="1450465951174" MODIFIED="1450466045416"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;If the constraint f_i is inactive (f_i(x*) &lt; 0), we have a_i = 0. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>a_i &gt; 0 is only possible if the constraint is active (f_i(x*)=0).</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
</node>
<node TEXT="Support Vector Machines" STYLE_REF="Stichpunkt" FOLDED="true" POSITION="right" ID="ID_1162915396" CREATED="1450534234009" MODIFIED="1450534242307">
<node TEXT="Hyperplanes in HNF" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_551877692" CREATED="1450535318716" MODIFIED="1450535468329">
<hook EQUATION="w^Tx+b=0" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="Far side" STYLE_REF="Beschreibung" ID="ID_1434667371" CREATED="1450535421340" MODIFIED="1459700892700">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1784421708" STARTINCLINATION="206;0;" ENDINCLINATION="206;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook EQUATION="w^Tx_1+b&gt;0" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Linear classifier" STYLE_REF="Beschreibung" ID="ID_1784421708" CREATED="1450535701193" MODIFIED="1459700903603" HGAP_QUANTITY="116.0 px" VSHIFT_QUANTITY="-8.0 px">
<hook EQUATION="h(x)=sgn(w^Tx+b)\\&#xa;sgn(z)=\begin{cases}&amp;-1 &amp;&amp;\text{if } z&lt;0\\&#xa;&amp;0 &amp;&amp;\text{if }\; z=0\\&#xa;&amp;+1 &amp;&amp;\text{if }\; z&gt;0\end{cases}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="FreeNode"/>
</node>
</node>
<node TEXT="Near side" STYLE_REF="Beschreibung" ID="ID_1310380866" CREATED="1450535534342" MODIFIED="1450537277313">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1784421708" STARTINCLINATION="206;0;" ENDINCLINATION="206;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook EQUATION="w^Tx_2+b&lt;0" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Hyperplane with margin" STYLE_REF="Beschreibung" ID="ID_334154093" CREATED="1450535928347" MODIFIED="1459701683193">
<hook EQUATION="w^Tx+(b-s)&gt;0\\&#xa;w^Tx+(b+s)&lt;0\\&#xa;\Downarrow\\&#xa;\begin{align}&amp;w^Tx_i+b\geq +1&amp;&amp;\text{for } y_i=+1,\\&#xa;&amp;w^Tx_i+b\leq -1 &amp;&amp;\text{for } y_i=-1\end{align}\\&#xa;\Downarrow\\&#xa;y_i (w^Tx_i+b)\geq 1 \;\text{for all $i$ with }y\in\{-1,1\}" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="Orthogonal distance to origin" STYLE_REF="Beschreibung" ID="ID_1945073078" CREATED="1450536571930" MODIFIED="1459701483563">
<hook EQUATION="d=-\frac b {|w|},\;d_{blue}=-\frac{b-s}{|w|},\;d_{green}=-\frac {b+s}{|w|}\\&#xa;m = d_{blue} - d_{green}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Margin" STYLE_REF="Beschreibung" ID="ID_1036230988" CREATED="1450536672459" MODIFIED="1450536981520"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;The size of the margin depends on s and |w|. Without constraining the problem we can set s=1.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="m=\frac{2s}{|w|}\Rightarrow m=\frac 2 {|w|} = \frac 2 {\sqrt{w^Tw}}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Optimization problem" STYLE_REF="Beschreibung" ID="ID_243979836" CREATED="1450537326597" MODIFIED="1459700865020">
<hook EQUATION="\begin{align}&amp;minimize&amp;&amp;f_0(w,b)=\frac 1 2 w^Tw\\&#xa;&amp;subject\;to &amp;&amp;f_i(w,b)=y_i(w^Tx_i+b)-1\geq 0,\;i=1,\ldots,m\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Factor 1/2 was introduced in the objective function to obtain nicer derivatives later on.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="1. Calculate the Lagrangian" STYLE_REF="Beschreibung" ID="ID_1443970741" CREATED="1450538199866" MODIFIED="1450538250146">
<hook EQUATION="L(w,b,\alpha)=\frac 1 2 w^Tw - \sum_{i=1}^m \alpha_i[y_i(w^Tx_i+b)-1]" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="2. Minimize L w.r.t. w and b" STYLE_REF="Beschreibung" ID="ID_476875275" CREATED="1450538254081" MODIFIED="1459700872078">
<hook EQUATION="\nabla_w L(w,b,\alpha)=w-\sum_{i=1}^m \alpha_i y_i x_i = 0\\&#xa;\frac{\partial L}{\partial b} = \sum_{i=1}^m \alpha_i y_i = 0" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Weights" STYLE_REF="Beschreibung" ID="ID_675473131" CREATED="1450538355498" MODIFIED="1450538397939"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Thus the weights are a linear combination of the training samples</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="w = \sum_{i=1}^m\alpha_i y_i x_i" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="3. Dual problem" STYLE_REF="Beschreibung" ID="ID_1341348074" CREATED="1450538448348" MODIFIED="1459700874707">
<hook EQUATION="\begin{align}&amp;maximize &amp;&amp;g(\alpha)=\sum_{i=1}^m \alpha_i - \frac 1 2 \sum_{i=1}^m\sum_{j=1}^my_iy_j\alpha_i\alpha_jx_i^Tx_j\\&#xa;&amp;subject\;to&amp;&amp;\sum_{i=1}^m\alpha_iy_i=0\\&#xa;&amp; &amp;&amp;\alpha_i\geq 0,\; i=1,\ldots,m\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;The bias b does not appear in the dual problem and thus needs to be recovered afterwards.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Support vectors" STYLE_REF="Stichpunkt" ID="ID_1733461048" CREATED="1450538729880" MODIFIED="1450538879219"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Remember that the optimal solution must satisfy amongst others the KKT complementary slackness condition. Hence a training sample x_i can only contribute to the weight vector if it lies on the margin. Such a training sample is called support vector.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\text{Complementary slackness: } \alpha_i \neq 0\\&#xa;y_i(w^Tx_i+b)=1" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Classifying" STYLE_REF="Stichpunkt" ID="ID_130415808" CREATED="1450538948796" MODIFIED="1513528641846">
<hook EQUATION="h(x)=sgn\left( \underbrace{\sum_{i=1}^m \alpha_iy_ix_i^Tx +b}_{w^Tx + b} \right)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Recovering the bias" STYLE_REF="Stichpunkt" ID="ID_1267086521" CREATED="1459702380245" MODIFIED="1459702435891">
<hook EQUATION="w^Tx_i + b= y_i" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;We can use any support vector to calculate the bias.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
</node>
<node TEXT="Dealing with outliers" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1689885655" CREATED="1450559510968" MODIFIED="1450559689358"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Introduce slack variable.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\xi_i \geq 0\\&#xa;w^Tx_i+b\geq+1-\xi_i\;\text{for }y_i=+1\\&#xa;w^Tx_i+b\leq-1+\xi_i\;\text{for }y_i=-1\\&#xa;\Downarrow\\&#xa;y_i(w^Tx_i+b)\geq 1-\xi_i\;\text{for all }i" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="Optimization problem with slack variables" STYLE_REF="Beschreibung" ID="ID_576151735" CREATED="1450559910518" MODIFIED="1459702471064">
<hook EQUATION="\begin{align}&amp;minimize&amp;&amp;f_0(w,b,\xi)=\frac 1 2 w^T w+C\sum_{i=1}^m\xi_i&amp;\\&#xa;&amp;subject\;to&amp;&amp;y_i(w^Tx_i+b)-1+\xi_i\geq0,&amp;i=1,\ldots,m\\&#xa;&amp; &amp;&amp;\xi_i\geq0&amp;i=1,\ldots,m\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;Here we used the 1-norm for the penalty term. Another choice is to use the 2-norm penalty. The penalty that performs better in practise will depend on the data and the type of noise that has influenced it.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Factor C" STYLE_REF="Beschreibung" ID="ID_1021586928" CREATED="1450560070638" MODIFIED="1450560119028"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="#666666" size="1"><i>&#160;The factor C &gt; 0 determines how heavy a violation punished. </i></font>
    </p>
    <p>
      <font color="#666666" size="1"><i>C -&gt; inf recovers the SVM without slack variables.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="1. Calculate Lagrangian" STYLE_REF="Beschreibung" ID="ID_1523135863" CREATED="1450560273202" MODIFIED="1450560424601">
<hook EQUATION="\begin{align}&#xa;L(w,b,\xi,\alpha,\mu)=&amp;\frac 1 2 w^Tw+C\sum_{i=1}^m\xi_i\\&#xa;&amp;-\sum_{i=1}^m\alpha_i[y_i(w^Tx_i+b)-1+\xi_i]-\sum_{i=1}^m\mu_i\xi_i&#xa;\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="2. Minimize L w.r.t. w,b and xi" STYLE_REF="Beschreibung" ID="ID_192275054" CREATED="1450560429266" MODIFIED="1459702473327">
<hook EQUATION="\nabla_wL(w,b,\xi,\alpha,\mu)=w-\sum_{i=1}^m\alpha_iy_ix_i=0\\&#xa;\frac{\partial L}{\partial b}=\sum_{i=1}^m\alpha_iy_i=0\\&#xa;\frac{\partial L}{\partial\xi_i}=C-\alpha_i-\mu_i=0,\;i=1,\ldots,m" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_1863957759" CREATED="1450560577865" MODIFIED="1450560734217">
<hook EQUATION="\text{with }\alpha_i = C-\mu,\;\mu_i\geq0,\;\alpha_i\geq0\\&#xa;\Rightarrow 0\geq\alpha_i\geq C" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="3. Dual problem with slack variables" STYLE_REF="Beschreibung" ID="ID_813476738" CREATED="1450560761962" MODIFIED="1459702481361">
<hook EQUATION="\begin{align}&amp;maximize &amp;&amp;g(\alpha)=\sum_{i=1}^m \alpha_i - \frac 1 2 \sum_{i=1}^m\sum_{j=1}^my_iy_j\alpha_i\alpha_jx_i^Tx_j\\&#xa;&amp;subject\;to&amp;&amp;\sum_{i=1}^m\alpha_iy_i=0\\&#xa;&amp; &amp;&amp;0\leq\alpha_i\underbrace{\leq C}_{new},\; i=1,\ldots,m\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="KKT conditions with slack variables" STYLE_REF="Beschreibung" ID="ID_227358072" CREATED="1450560980412" MODIFIED="1459703103368">
<hook EQUATION="\begin{align}&amp;y_i(wx_i+b)-1+\xi_i\geq0&amp;&amp;\text{primal feasibility}&amp;(1)\\&#xa;&amp;\xi_i\geq0&amp;&amp;\text{primal feasibility}&amp;(2)\\&#xa;&amp;0\leq\alpha_i\leq C&amp;&amp;\text{dual feasibility for }\alpha,\mu&amp;\\&#xa;&amp;\alpha_i[y_i(wx_i+b)-1+\xi_i]=0&amp;&amp;\text{complementary slackness for }\alpha_i&amp;(3)\\&#xa;&amp;\xi_i(C-\alpha)=0&amp;&amp;\text{complementary slackness for }\mu_i&amp;(4)\end{align}\\&#xa;\frac{\partial L}{\partial w}=0\\&#xa;\frac{\partial L}{\partial b}=0" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;These are necessary and sufficient conditions for a solution to be optimal.</i></font>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="alpha_i = 0" STYLE_REF="Beschreibung" ID="ID_1258320884" CREATED="1450561327352" MODIFIED="1450562043279">
<hook EQUATION="\alpha_i=0:&#xa;\begin{align}&#xa;(4)\rightarrow  &amp;\xi_i=0\\&#xa;(1)\rightarrow  &amp;x_i \text{ correctly classified but not a support vector.}\\&#xa;\; &amp;\text{It may or may not lie on the boundary of the margin.}&#xa;\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="0 &lt; alpha_i &lt; C" STYLE_REF="Beschreibung" ID="ID_1861992890" CREATED="1450562123763" MODIFIED="1450562304857">
<hook EQUATION="0&lt;\alpha_i&lt;C:&#xa;\begin{align}&#xa;(4) \rightarrow &amp; \xi_i=0\\&#xa;(3) \rightarrow &amp; x_i \text{ correctly classified and a support vector.}\\&#xa;&amp; \text{It lies on the boundary of the margin.}&#xa;\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="alpha_i = C" STYLE_REF="Beschreibung" ID="ID_1282535355" CREATED="1450562316563" MODIFIED="1450562633954">
<hook EQUATION="\alpha_i=C:&#xa;\begin{align}&#xa;(2),(3)\rightarrow &amp;y_i(w^Tx_i+b)-1\leq0\\&#xa;&amp; x_i\text{ can violate the margin and is possibly misclassified}&#xa;\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Simpliefied KKT conditions" STYLE_REF="Beschreibung" ID="ID_272587272" CREATED="1450562744881" MODIFIED="1450563421925">
<hook EQUATION="y_i \left( \sum_{j=1}^l y_j\alpha_jx_j^Tx_i+b \right)&#xa;\begin{cases}&#xa;\geq 1     &amp;&amp; \text{if } \alpha_i=0    &amp;    (\text{correctly classified})\\&#xa;=1         &amp;&amp; \text{if } 0&lt;\alpha_i&lt;C   &amp;   (\text{support vector})\\&#xa;\leq 1     &amp;&amp; \text{if } \alpha_i=C      &amp;  (\text{violating margin})&#xa;\end{cases}\\&#xa;\sum_{i=1}^ly_i\alpha_i=0" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Recovering the bias" STYLE_REF="Beschreibung" ID="ID_537893262" CREATED="1450563553659" MODIFIED="1450563621126">
<hook EQUATION="b = y_i - \sum_{j=1}^ly_j\alpha_jx_j^Tx_i" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Given any support vector x_i the bias b is given by</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
</node>
<node TEXT="Sequential Minimal Optimization (SMO)" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1362126613" CREATED="1450566081933" MODIFIED="1450569550736">
<hook EQUATION="\begin{align}&amp;maximize &amp;&amp;g(\alpha)=\sum_{i=1}^m \alpha_i - \frac 1 2 \sum_{i=1}^m\sum_{j=1}^my_iy_j\alpha_i\alpha_jx_i^Tx_j&amp;\\&#xa;&amp;subject\;to&amp;&amp;\sum_{i=1}^m\alpha_iy_i=0&amp;(1)\\&#xa;&amp; &amp;&amp;0\leq\alpha_i\leq C,\; i=1,\ldots,m&amp;(2)\end{align}\\" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">1 initialize all &#945;is to zero </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">2 </font></i><font color="#666666" size="1"><b>while</b><i>&#160;g(&#945;) not converged </i><b>do </b></font>
    </p>
    <p>
      <i><font color="#666666" size="1">3&#160;&#160;&#160;&#160;&#160;&#160;choose a training sample xk that violates the </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;KKT conditions </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">4&#160;&#160;&#160;&#160;&#160;&#160;choose a second training sample xl so that </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">&#160;&#160;&#160;&#160;&#160;&#160;&#160;|Ek &#8722; El | is maximized </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">5&#160;&#160;&#160;&#160;&#160;&#160;&#945;l &#8592; &#945; new l </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">6&#160;&#160;&#160;&#160;&#160;&#160;&#945;k &#8592; &#945; old k + yk yl (&#945; old l &#8722; &#945; new l ) </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">7 </font></i><font color="#666666" size="1"><b>end</b></font>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="Componentwise optimisation" STYLE_REF="Beschreibung" ID="ID_1418777462" CREATED="1450567204174" MODIFIED="1450567277106"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Since g is concave, we can find the global maximum by iteratively maximizing (or just moving towards the maximum the components of alpha independently.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Keeping constraints satisfied" STYLE_REF="Beschreibung" ID="ID_67172916" CREATED="1450567513038" MODIFIED="1459711153349"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Let us assume that we have a parameter vector alpha that satisfies constraint (1). We must update at least two components of alpha at once to ensure that the constraint stays satisfied. They must satisfy</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\alpha_ky_k+\alpha_ly_l = \alpha_k^{old}y_k+\alpha_l^{old}y_l\\&#xa;\Rightarrow \alpha_k=\gamma-s\alpha_l\\&#xa;\text{with }s=y_ky_l,\;\gamma=\alpha_k^{old}+s\alpha_l^{old}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="..." STYLE_REF="Beschreibung" ID="ID_510824850" CREATED="1450567840556" MODIFIED="1450568125911"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;The constraint (2) yields a feasible range for alpha_l</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="U\leq\alpha_l\leq V,\\&#xa;U=\begin{cases}&#xa;\max(0,\alpha_l^{old}-\alpha_k^{old})&amp;&amp;\text{if }y_k\neq y_l\\&#xa;\max(0,\alpha_k^{old}+\alpha_l^{old}-C)&amp;&amp;\text{if }y_k=y_l\end{cases}\\&#xa;V=\begin{cases}&#xa;\min(C,C-\alpha_k^{old}+\alpha_l^{old})&amp;&amp;\text{if }y_k \neq y_l\\&#xa;\min(C,\alpha_k^{old}+\alpha_l^{old})&amp;&amp;\text{if }y_k=y_l\end{cases}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Analytical solution for two components" STYLE_REF="Beschreibung" ID="ID_1903370126" CREATED="1450568208302" MODIFIED="1459711155323"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Consider the objective as a function of alpha_k and alpha_l (treat other alphas as constants)</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\begin{align}&#xa;g(\alpha_k,\alpha_l)=&amp;\alpha_k+\alpha_l-\frac 1 2 x_k^2\alpha_k^2 - \frac 1 2 x_l^2\alpha_l^2\\&#xa;&amp;-y_ky_lx_k^Tx_l\alpha_k\alpha_l-y_k\alpha_kv_k-y_l\alpha_lv_l+const\\&#xa;\text{with constant }v_n=\sum_{\substack{j=1\\j\neq k\\j\neq l}}^my_j\alpha_jx_n^Tx_j\end{align}\\&#xa;\Rightarrow (\alpha_k=\gamma-s\alpha_l),\;\frac{dg}{d\alpha_l}=0" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Maximum" STYLE_REF="Beschreibung" ID="ID_1448116319" CREATED="1450568701013" MODIFIED="1450568962714">
<hook EQUATION="\alpha_l^{max} = \alpha_l^{old} + \frac{y_k(E_k-E_l)}{x_k^2-2x_k^Tx_l+x_l^2}\\&#xa;\text{with }E_n=f(x_n)-y_n\\&#xa;a_l^{new}=\begin{cases}&#xa;U&amp;&amp;\text{if }\alpha_l^{max}&lt;U\\&#xa;\alpha_l^{max}&amp;&amp;\text{if }U\leq\alpha_l^{max}\leq V\\&#xa;V&amp;&amp;\text{if }V\leq\alpha_l^{max}\end{cases}" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;We can safely do this because g is a concave function.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Kernels" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1170079672" CREATED="1450570096235" MODIFIED="1450570337058"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;So far we can only construct linear classifiers. To overcome this limitation we map the training samples into a high dimensional feature space with the corresponding kernel (scalar product).</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="x_i\rightarrow \phi(x_i),\;K(x_i,x_j)=\phi(x_i)^T\phi(x_j)\\&#xa;g(\alpha)=\sum_{i=1}^m\alpha_i - \frac 1 2 \sum_{i=1}^m\sum_{j=1}^my_iy_j\alpha_i\alpha_jx_i^Tx_j\\&#xa;\text{replace }x_i^Tx_j\text{ by }K(x_i,x_j)" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="Polynomial kernel" STYLE_REF="Beschreibung" ID="ID_1966695144" CREATED="1450570361950" MODIFIED="1450570428118">
<hook EQUATION="K(a,b)=(a^Tb)^p\text{ or }(a^Tb+1)^p" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Gaussian kernel" STYLE_REF="Beschreibung" ID="ID_1519328105" CREATED="1450570418809" MODIFIED="1459713257947">
<hook EQUATION="K(a,b)=\exp\left( - \frac{\| a-b \|^2}{2\sigma^2} \right)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="How to choose parameters C and &#x3c3;?" STYLE_REF="Beschreibung" ID="ID_613912919" CREATED="1459713186645" MODIFIED="1459713251744"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;Grid search for optimal parameters using cross-validation.</i></font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Sigmoid" STYLE_REF="Beschreibung" ID="ID_1707729096" CREATED="1450570463258" MODIFIED="1450570502474">
<hook EQUATION="K(a,b)=\tanh(\kappa a^Tb-\delta)\text{ for }\kappa,\delta&gt;0" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Multiple classes" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1055153275" CREATED="1450570655031" MODIFIED="1450570702646"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Since SVMs can only do binary classification, multiple classes need some tricks:</font></i>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="one-vs-rest" STYLE_REF="Beschreibung" ID="ID_1412461302" CREATED="1450570705734" MODIFIED="1450570747173"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Train n SVMs for n classes, where each SVM is being trained for classification of one class against all the remaining ones. The winner is then the class, where the distance from the hyperplane is maximal. </font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="one-vs-one" STYLE_REF="Beschreibung" ID="ID_427076195" CREATED="1450570748562" MODIFIED="1450570894889"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Train n(n &#8722; 1)/2 classes (all possible pairings) and evaluate all. The winner class it the class which wins most. </font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Gaussian Process Regression" STYLE_REF="Stichpunkt" FOLDED="true" POSITION="right" ID="ID_1578267999" CREATED="1450610753580" MODIFIED="1450610776963">
<node TEXT="Distribution over vectors" STYLE_REF="Beschreibung" ID="ID_580110133" CREATED="1450610961022" MODIFIED="1450610981246">
<hook EQUATION="w\sim \mathcal N (\mu,\Sigma)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Distribution over functions" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1321144857" CREATED="1450610984402" MODIFIED="1450611388817">
<hook EQUATION="f\sim \mathcal G \mathcal P (m,K)" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Our basic assumption is that if vectors x and x' are similar, then f(x) and f(x') should be similar, too.</font></i>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="Mean function" STYLE_REF="Beschreibung" ID="ID_1949220257" CREATED="1450611042227" MODIFIED="1459720923469">
<hook EQUATION="m:\mathcal X\rightarrow \mathbb R\\&#xa;m(x)=\mathbb E [f(x)]" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;The mean function m(x) encodes the a priori expectation of the (unknown) function.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Setting the mean function" STYLE_REF="Beschreibung" ID="ID_1163970694" CREATED="1450611550186" MODIFIED="1450620928610"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;In most cases we simply use zero expectation, which makes sense especially if we normalise the output to zero mean.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\mathbb E[f(x)]=m(x)=0" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Covariance function" STYLE_REF="Beschreibung" ID="ID_894802167" CREATED="1450611080728" MODIFIED="1459720926141">
<hook EQUATION="K:\mathcal X^2 \rightarrow \mathbb R\\&#xa;K(x,x&apos;) = \mathbb E [(f(x)-m(x))(f(x&apos;)-m(x&apos;))]" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;The covariance function K returns a measure of the similarity of x and x' and encodes how similar f(x) and f(x') should be.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Setting the covariance function" STYLE_REF="Beschreibung" ID="ID_1264273726" CREATED="1450611643100" MODIFIED="1459720930462"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;The &quot;default&quot; covariance function is squared exponential kernel, where l varies the length (or width) and sigma the height of the kernel. Note that x can be in any domain, if K defines a good measure of the similarity of two vectors x and x'. The covariance function is what drives the behaviour of a GP!</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="K(x,x&apos;)=\sigma^2 \exp \left( -\frac{(x-x&apos;)^T(x-x&apos;)}{2l^2} \right)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Similar points" STYLE_REF="Beschreibung" ID="ID_1857738287" CREATED="1450612876917" MODIFIED="1450612947739"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;If two points are similar, they covary strongly. Knowing about f1 reveals a lot about f2.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Points far apart" STYLE_REF="Beschreibung" ID="ID_1343781153" CREATED="1450612948426" MODIFIED="1450613000156"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;If two points are far apart, their covariace is small. Knowing the value of f1 reveals little about f4.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Extreme case" STYLE_REF="Beschreibung" ID="ID_518325850" CREATED="1450613014095" MODIFIED="1450613047006"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;In the extreme case where the covariance is 0, the conditional and marginal distributions are the same.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Formal properties of the covariance function" STYLE_REF="Beschreibung" ID="ID_1419317555" CREATED="1450616916438" MODIFIED="1459720932709"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;The covariance function K needs to be a measure of similarity between x and x'. This is the basic assumption which makes inference possible, since we assume that similar data points have similar function values.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="K needs to be symmetric" STYLE_REF="Beschreibung" ID="ID_1593273789" CREATED="1450616992162" MODIFIED="1450617013756">
<hook EQUATION="K(x,x&apos;)=K(x&apos;,x)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="K needs to be positive semidefinite" STYLE_REF="Beschreibung" ID="ID_979595162" CREATED="1450617014997" MODIFIED="1450617024137"/>
</node>
</node>
<node TEXT="Inference" STYLE_REF="Beschreibung" ID="ID_1409596458" CREATED="1450611458334" MODIFIED="1450611499978"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;For inference we condition the unknown function values on the known ones. If there are no &quot;similar&quot; known values, then the mean function dominates the result.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Gaussian process" STYLE_REF="Stichpunkt" FOLDED="true" ID="ID_1885432415" CREATED="1450611961063" MODIFIED="1450612482500"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;A Gaussian process is a collection of random variables (RV), any finite number of which have joint Gaussian distribution.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="f(X)\sim\mathcal N(m(X),K(X,X))\\&#xa;\text{where }m(X)=(m(x_1),\ldots,m(x_n))^T (\text{often }m(X)=0)\\&#xa;\text{and }K(X,X)=\begin{pmatrix}&#xa;K(x_1,x_1) &amp;K(x_1,x_2) &amp;\cdots &amp;K(x_1,x_n)\\&#xa;K(x_2,x_1 &amp;K(x_2,x_2)) &amp;\cdots &amp;K(x_2,x_n)\\&#xa;\vdots &amp; \vdots &amp;\vdots &amp;\vdots\\&#xa;K(x_n,x_1) &amp;K(x_n,x_2) &amp;\ldots &amp;K(x_n,x_n)\end{pmatrix}" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="Drawing samples from the prior" STYLE_REF="Beschreibung" ID="ID_914084080" CREATED="1450613324901" MODIFIED="1450613423750"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Sampling from the prior distribution of a GP at arbitrary points X* is equivalent to sampling from an MVN:</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="f_{pri}(X_*)\sim\mathcal N(m(X_*),K(X_*,X_*))" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Gaussian process regression" STYLE_REF="Stichpunkt" FOLDED="true" ID="ID_211089268" CREATED="1450613529041" MODIFIED="1450613905630"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;We have training data X (of size N x D), corresponding observations f=f(X), and test data points X* (N* x D) for which we want to infer function values f*=f(X*). </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">The GP defines a joint distribution for p(f,f* | X,X*):</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\begin{pmatrix}f\\f_*\end{pmatrix}\sim\mathcal N\left(&#xa;\begin{bmatrix}\mu\\ \mu_*\end{bmatrix},\begin{bmatrix}K&amp;K_*\\K_*^T&amp;K_{**}\end{bmatrix}&#xa;\right)\\&#xa;\text{with }\mu=m(X),\mu_*=m(X_*)\\&#xa;K=K(X,X),K_*=K(X,X_*),K_{**}=K(X_*,X_*)" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="Inference" STYLE_REF="Beschreibung" ID="ID_129365231" CREATED="1450613930242" MODIFIED="1459755546451"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;To infer f* or rather p(f* | X*,X,f), we need to apply the rules for conditioning multivariate Gaussians.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="f_*|f,X,X_* \sim \mathcal N\left(\mu_*+K_*^TK^{-1}(f-\mu),K_{**}-K_*^TK^{-1}K_*\right)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Conditionals of an MVN" STYLE_REF="Beschreibung" ID="ID_488002534" CREATED="1450614380917" MODIFIED="1450614693723">
<hook EQUATION="y=\begin{bmatrix}y_1\\y_2\end{bmatrix} \sim \mathcal N \left( &#xa;y=\begin{bmatrix}\mu_1\\\mu_2\end{bmatrix},\Sigma=\begin{bmatrix}&#xa;\Sigma_{11}&amp;\Sigma_{12}\\\Sigma_{21}&amp;\Sigma_{22}\end{bmatrix} \right)\\&#xa;p(y_2|y_1): y_2|y_1 \sim \mathcal N (\mu_{2|1},\Sigma_{2|1})\\&#xa;\mu_{2|1}=\mu_2+\Sigma_{21}\Sigma_{11}^{-1}(y_1-\mu_1)\\&#xa;\Sigma_{2|1}=\Sigma_{22} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Inference in the noisy case" STYLE_REF="Beschreibung" ID="ID_850558219" CREATED="1450615538195" MODIFIED="1459755523805"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;In the noise-free case (with some kernels, such as the SE kernel) the GP acts as an interpolator between observed values. </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">More often than not, the assumption that our observations y_i correspond exactly to the function values f(x_i)=f_i is wrong. </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">We will now instead assume, that we observe a noisy version of the underlying function:</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="y_i = f_i + \epsilon,\\&#xa;\text{where } \epsilon\sim\mathcal N(0,\sigma_y^2)\text{ is additive iid Gaussian noise.}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Example 1" STYLE_REF="Beschreibung" ID="ID_474558339" CREATED="1450615799894" MODIFIED="1459755533855">
<hook EQUATION="m(x)=0, K(x,x&apos;)=1\text{ if }x=x&apos;\\&#xa;\text{In the noise-free case, where }y_i=f_i:\\&#xa;\begin{pmatrix}y_i\\f_i\end{pmatrix}\sim\mathcal N\left( 0,\begin{pmatrix}1&amp;1\\1&amp;1\end{pmatrix} \right),\\&#xa;\text{then }y_i|f_i \sim \mathcal N(f_i,0)\text{ is a degenerate Gaussian with zero variance.}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Noisy scenario" STYLE_REF="Beschreibung" ID="ID_387760039" CREATED="1450616115645" MODIFIED="1450616255885">
<hook EQUATION="\text{We want }y_i|f_i\sim\mathcal N(f_i,\sigma_y^2), \text{ so we assume instead:}\\&#xa;\begin{pmatrix}y_i\\f_i\end{pmatrix}\sim\mathcal N \left(&#xa;0, \begin{pmatrix}1+\sigma_y^2&amp;1\\1&amp;1\end{pmatrix}\right)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Arbitrary X*" STYLE_REF="Beschreibung" ID="ID_1271664990" CREATED="1450616314964" MODIFIED="1459755529077"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;We can easily extend this idea for arbitrary X*. Since the individual noise terms are independent we have to add a scaled identity matrix.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\begin{bmatrix}y\\f(X_*)=f_*\end{bmatrix}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Joint distribution in the noisy case" STYLE_REF="Beschreibung" ID="ID_673398812" CREATED="1450616441492" MODIFIED="1450616543107">
<hook EQUATION="\begin{bmatrix}y\\f_*\end{bmatrix}\sim\mathcal N\left(  &#xa;\begin{bmatrix}\mu\\\mu_*\end{bmatrix},\begin{bmatrix}K+\sigma_y^2I&amp;K_*\\K_*^T&amp;K_{**}\end{bmatrix}\right)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Conditional (predictive) distribution" STYLE_REF="Beschreibung" ID="ID_1082677982" CREATED="1450616545948" MODIFIED="1450616664745">
<hook EQUATION="f_*|y,X,X_* \sim\mathcal N (\mu_*+K_*^T[K+\sigma_y^2I]^{-1}(f-\mu)\;,\; K_{**}-K_*^T[K+\sigma_y^2]^{-1}K_*)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Point prediction" STYLE_REF="Beschreibung" ID="ID_495094070" CREATED="1450616732486" MODIFIED="1450616874969"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Obviously, we will use the expectation as our point prediction:</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\hat f_*=\mathbb E[f_*|X,y,X_*] = \mu_*+K_*^T[K+\sigma_y^2I]^{-1}(f-\mu)\\&#xa;cov(f_*)=K_{**}-K_*^T[K+\sigma_y^2]^{-1}K_*" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
</node>
<node TEXT="Pros and Cons of GPs" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_220523870" CREATED="1450617190459" MODIFIED="1450617195934">
<node TEXT="Pros" STYLE_REF="Beschreibung" ID="ID_1591048508" CREATED="1450617199972" MODIFIED="1459755567687">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="+ GPs can fit arbitrary functions" STYLE_REF="Beschreibung" ID="ID_237313487" CREATED="1450617211673" MODIFIED="1450617356777"/>
<node TEXT="+ Probabilistic model: returns measure of uncertainty" STYLE_REF="Beschreibung" ID="ID_597446151" CREATED="1450617227900" MODIFIED="1450617357204"/>
<node TEXT="+ Domain knowledge can be included through design of the mean and covariance functions" STYLE_REF="Beschreibung" ID="ID_3324810" CREATED="1450617242534" MODIFIED="1450617357589"/>
<node TEXT="+ GPs can be employed for regression and classification (out of scope)" STYLE_REF="Beschreibung" ID="ID_1919727527" CREATED="1450617262526" MODIFIED="1450617357948"/>
<node TEXT="+ Handles different types of input well (as long as an appropriate K is used)" STYLE_REF="Beschreibung" ID="ID_374564319" CREATED="1450617280707" MODIFIED="1450617358319"/>
<node TEXT="+ Hyper parameters (e.g. parameters of the covariance function, noise) can be learned by maximising the marginal likelihood (not covered here)" STYLE_REF="Beschreibung" ID="ID_108030675" CREATED="1450617305914" MODIFIED="1450617358842"/>
</node>
<node TEXT="Cons" STYLE_REF="Beschreibung" ID="ID_1184483965" CREATED="1450617204460" MODIFIED="1459755565326">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="- The inversion of the kernel matrix limits the training set to ca. 1000 samples (-&gt; fixed by sparse GP)" STYLE_REF="Beschreibung" ID="ID_1097215735" CREATED="1450617363698" MODIFIED="1450617439588"/>
<node TEXT="- All training samples are stored explicitly (-&gt; fixed by sparse GP)" STYLE_REF="Beschreibung" ID="ID_1558395567" CREATED="1450617395149" MODIFIED="1450617439285"/>
<node TEXT="- The success depends strongly on the choice of covariance function." STYLE_REF="Beschreibung" ID="ID_257510197" CREATED="1450617421069" MODIFIED="1450617438711"/>
</node>
</node>
<node TEXT="GP is a non-parametric model" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_675195165" CREATED="1450617043393" MODIFIED="1450617084652">
<node TEXT="Parametric model" STYLE_REF="Beschreibung" ID="ID_1185610907" CREATED="1450617088026" MODIFIED="1450617124408"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;The number of parameters in a parametric model is fixed before training, while in non-parametric models it grows with the number of training samples.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Non-parametric model" STYLE_REF="Beschreibung" ID="ID_717488278" CREATED="1450617125180" MODIFIED="1450617181939"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Require no or little training (e.g. kNN), while for parametric models (e.g. Linear Regression) training is typically more expensive than inference.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Neural Networks" STYLE_REF="Stichpunkt" FOLDED="true" POSITION="right" ID="ID_1756438974" CREATED="1452606512355" MODIFIED="1452606517560">
<node TEXT="Linear Regression Model" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1429908806" CREATED="1452607257322" MODIFIED="1452680184242"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;We have input vectors x and associated output values z. We want to describe the underlying functional relation.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="y(x,w) = w_0 + \sum_{j=1}^{M-1} w_j \phi_j(x)=w^T \phi (x)" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="Typical&#xa;Basis&#xa;Functions" STYLE_REF="Beschreibung" ID="ID_899854945" CREATED="1452607405434" MODIFIED="1459767280112">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Polynomials" STYLE_REF="Beschreibung" ID="ID_955140900" CREATED="1452607416033" MODIFIED="1452607475745">
<hook EQUATION="1,x,x^2,x^3,\ldots" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Gaussians" STYLE_REF="Beschreibung" ID="ID_99223898" CREATED="1452607419922" MODIFIED="1452607493257">
<hook EQUATION="e^{-(x-\mu)^2}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Sigmoids" STYLE_REF="Beschreibung" ID="ID_1391382088" CREATED="1452607422714" MODIFIED="1452607536221">
<hook EQUATION="\frac{1}{1+e^{-(x-\mu)}},\;\tanh(x-\mu)" NAME="plugins/latex/LatexNodeHook.properties"/>
<font SIZE="8"/>
</node>
</node>
<node TEXT="Schematic representation" STYLE_REF="Beschreibung" ID="ID_839603257" CREATED="1452607573870" MODIFIED="1459767282976">
<hook EQUATION="y(x,w)=w^T\phi(x)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="y" STYLE_REF="Beschreibung" ID="ID_87561464" CREATED="1452607668920" MODIFIED="1481645589421">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="x" STYLE_REF="Beschreibung" ID="ID_600695383" CREATED="1452607675035" MODIFIED="1452608275220" HGAP_QUANTITY="52.0 px"/>
<node TEXT="x" STYLE_REF="Beschreibung" ID="ID_351683976" CREATED="1452607684690" MODIFIED="1452608274905" HGAP_QUANTITY="52.0 px"/>
<node TEXT="x" STYLE_REF="Beschreibung" ID="ID_225809298" CREATED="1452607685649" MODIFIED="1452608274548" HGAP_QUANTITY="52.0 px"/>
<node TEXT="x" STYLE_REF="Beschreibung" ID="ID_1951551783" CREATED="1452607686409" MODIFIED="1452608274187" HGAP_QUANTITY="52.0 px" VSHIFT_QUANTITY="-1.0 px"/>
<node TEXT="x" STYLE_REF="Beschreibung" ID="ID_31100021" CREATED="1452607687768" MODIFIED="1452608273806" HGAP_QUANTITY="52.0 px" VSHIFT_QUANTITY="-5.0 px"/>
</node>
</node>
<node TEXT="Extend system with additional layer" STYLE_REF="Beschreibung" ID="ID_1450030087" CREATED="1452607939972" MODIFIED="1459767290937">
<hook EQUATION="y(x,w_0,w_1)=w_1^T\phi(w_0\,x)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="y" STYLE_REF="Beschreibung" ID="ID_1615957002" CREATED="1452608023405" MODIFIED="1459767294362">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_1146344403" CREATED="1452608026160" MODIFIED="1452608279430" HGAP_QUANTITY="60.0 px" VSHIFT_QUANTITY="6.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1376130192" STARTINCLINATION="68;0;" ENDINCLINATION="68;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1858771436" STARTINCLINATION="91;0;" ENDINCLINATION="91;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
</node>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_451371839" CREATED="1452608035584" MODIFIED="1452608279061" HGAP_QUANTITY="60.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1376130192" STARTINCLINATION="68;0;" ENDINCLINATION="68;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1858771436" STARTINCLINATION="80;0;" ENDINCLINATION="80;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
</node>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_1434994820" CREATED="1452608036309" MODIFIED="1452608278706" HGAP_QUANTITY="60.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1858771436" STARTINCLINATION="74;0;" ENDINCLINATION="74;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1376130192" STARTINCLINATION="74;0;" ENDINCLINATION="74;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
</node>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_1665731527" CREATED="1452608041607" MODIFIED="1459767309562" HGAP_QUANTITY="60.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1858771436" STARTINCLINATION="74;0;" ENDINCLINATION="74;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="1" OBJECT="java.lang.Long|1" STYLE_REF="Beschreibung" ID="ID_1376130192" CREATED="1452608050700" MODIFIED="1452608313648" HGAP_QUANTITY="67.0 px" VSHIFT_QUANTITY="-51.0 px">
<hook NAME="FreeNode"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;for simplicity, the constant &quot;1&quot; is usually and from now on not depicted. But you always need it!</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_22777569" CREATED="1452608042939" MODIFIED="1459767303790" HGAP_QUANTITY="60.0 px" VSHIFT_QUANTITY="-1.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1376130192" STARTINCLINATION="100;0;" ENDINCLINATION="100;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="x" STYLE_REF="Beschreibung" ID="ID_1858771436" CREATED="1452608048111" MODIFIED="1452608321220" HGAP_QUANTITY="73.0 px" VSHIFT_QUANTITY="-21.0 px">
<hook NAME="FreeNode"/>
</node>
</node>
</node>
</node>
<node TEXT="Continue adding more hidden layers" STYLE_REF="Beschreibung" ID="ID_1043370820" CREATED="1452608343865" MODIFIED="1459767327042">
<hook EQUATION="y(x,w)=w_2^T\phi(w_1^T\phi(w_0\,x))" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="y" STYLE_REF="Beschreibung" ID="ID_370122369" CREATED="1452608359685" MODIFIED="1459767329156">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_346851526" CREATED="1452608364802" MODIFIED="1459767336336" HGAP_QUANTITY="58.0 px" VSHIFT_QUANTITY="12.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1207530826" STARTINCLINATION="64;0;" ENDINCLINATION="64;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_118060508" STARTINCLINATION="82;0;" ENDINCLINATION="82;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1286312829" STARTINCLINATION="72;0;" ENDINCLINATION="72;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_987247619" STARTINCLINATION="95;0;" ENDINCLINATION="95;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_424709731" CREATED="1452608433908" MODIFIED="1452608665989" HGAP_QUANTITY="45.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_613486647" STARTINCLINATION="76;0;" ENDINCLINATION="76;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
</node>
</node>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_973384956" CREATED="1452608369206" MODIFIED="1459767341332" HGAP_QUANTITY="61.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1207530826" STARTINCLINATION="59;0;" ENDINCLINATION="59;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_424709731" STARTINCLINATION="60;0;" ENDINCLINATION="60;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_118060508" STARTINCLINATION="68;0;" ENDINCLINATION="68;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_987247619" STARTINCLINATION="79;0;" ENDINCLINATION="79;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1286312829" STARTINCLINATION="62;0;" ENDINCLINATION="62;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_1207530826" CREATED="1452608432197" MODIFIED="1452608680483" HGAP_QUANTITY="45.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_613486647" STARTINCLINATION="66;0;" ENDINCLINATION="66;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
</node>
</node>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_1643705540" CREATED="1452608369975" MODIFIED="1459767343821" HGAP_QUANTITY="58.0 px" VSHIFT_QUANTITY="-1.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_118060508" STARTINCLINATION="62;0;" ENDINCLINATION="62;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1207530826" STARTINCLINATION="65;0;" ENDINCLINATION="65;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_424709731" STARTINCLINATION="70;0;" ENDINCLINATION="70;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_987247619" STARTINCLINATION="69;0;" ENDINCLINATION="69;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_1286312829" CREATED="1452608430054" MODIFIED="1459767351535" HGAP_QUANTITY="49.0 px" VSHIFT_QUANTITY="1.0 px">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="x" STYLE_REF="Beschreibung" ID="ID_613486647" CREATED="1452608639245" MODIFIED="1452608669381" HGAP_QUANTITY="52.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_118060508" STARTINCLINATION="69;0;" ENDINCLINATION="69;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
</node>
</node>
</node>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_67061060" CREATED="1452608370599" MODIFIED="1459767346378" HGAP_QUANTITY="58.0 px" VSHIFT_QUANTITY="1.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1207530826" STARTINCLINATION="74;0;" ENDINCLINATION="74;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1286312829" STARTINCLINATION="66;0;" ENDINCLINATION="66;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_424709731" STARTINCLINATION="84;0;" ENDINCLINATION="84;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_987247619" STARTINCLINATION="59;0;" ENDINCLINATION="59;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_118060508" CREATED="1452608427974" MODIFIED="1452608445180" HGAP_QUANTITY="45.0 px"/>
</node>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_636286568" CREATED="1452608371349" MODIFIED="1459767348809" HGAP_QUANTITY="58.0 px" VSHIFT_QUANTITY="-1.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1286312829" STARTINCLINATION="74;0;" ENDINCLINATION="74;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_118060508" STARTINCLINATION="62;0;" ENDINCLINATION="62;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_424709731" STARTINCLINATION="98;0;" ENDINCLINATION="98;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1207530826" STARTINCLINATION="85;0;" ENDINCLINATION="85;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_987247619" STARTINCLINATION="57;0;" ENDINCLINATION="57;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="bf" STYLE_REF="Beschreibung" ID="ID_987247619" CREATED="1452608424308" MODIFIED="1452608672188" HGAP_QUANTITY="43.0 px" VSHIFT_QUANTITY="1.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_613486647" STARTINCLINATION="79;0;" ENDINCLINATION="79;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
</node>
</node>
</node>
</node>
</node>
<node TEXT="How do we find W?" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_934004818" CREATED="1452608978307" MODIFIED="1452609232949"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Data consists of targets z and corresponding input vector X.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="z = y(x,w)+\epsilon\;\;\;[\epsilon:\text{ Gaussian, zero mean}]\\&#xa;\text{log likelihood: } \ln p(z|X,w)\propto - \frac 1 2 \sum_{n=1}^{N} (\left( z_n - y(x_n,w) \right))^2" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="Loss" STYLE_REF="Stichpunkt" ID="ID_1358929519" CREATED="1452609253752" MODIFIED="1459767910414"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Negative log likelihood</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\mathcal L(w)\text{ or }E(w)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="MLE solution" STYLE_REF="Stichpunkt" ID="ID_1507382256" CREATED="1452609331627" MODIFIED="1459767780101">
<hook EQUATION="\min\left(E = \sum_{n=1}^N \left( z_n-y(x_n,w) \right)^2 \right)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Optimization" STYLE_REF="Beschreibung" ID="ID_372610730" CREATED="1452609472371" MODIFIED="1452609749134"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;There is one difference w.r.t. linear regression: E(w) is <b>no longer convex</b>! </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">The minimum is located where its gradient is 0. So one typically minimises by using the gradient:</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="w_{i+1} = w_i - \alpha \nabla E" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="back-propagation" STYLE_REF="Stichpunkt" ID="ID_773689444" CREATED="1452609612673" MODIFIED="1459767782962"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;To compute the gradient, compute the residual y-z at the output, and propagate that back to the neurons in the layers below. From that you can then compute the gradient.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node ID="ID_1164706491" CREATED="1452609924759" MODIFIED="1452610239340"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1">algorithm</font>
    </p>
  </body>
</html>
</richcontent>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1">initialise the weights </font>
    </p>
    <p>
      <b><font size="1">repeat</font></b>
    </p>
    <p>
      <font size="1">&#160; <b>for</b>&#160;each training sample (x,z) do </font>
    </p>
    <p>
      <font size="1">&#160; <b>begin</b> </font>
    </p>
    <p>
      <font size="1">&#160;&#160;&#160;&#160;compute o=y(w,x) (forward pass) </font>
    </p>
    <p>
      <font size="1">&#160;&#160;&#160;&#160;calculate residual d_kj = z - o at the output units </font>
    </p>
    <p>
      <font size="1">&#160;&#160;&#160; <b>for</b>&#160;all k: </font>
    </p>
    <p>
      <font size="1">&#160;&#160;&#160;&#160;&#160;&#160;propagate d_kj back one layer by d_k-1,i = sum_j (d_kj w_k-1,i,j) </font>
    </p>
    <p>
      <font size="1">&#160;&#160;&#160;&#160;update the weights using grad E = d_kj phi'(.) x_i </font>
    </p>
    <p>
      <font size="1">&#160; <b>end</b> </font>
    </p>
    <p>
      <font size="1">&#160;&#160;(this is called one epoch ) </font>
    </p>
    <p>
      <b><font size="1">until </font></b><font size="1">stopping criterion satisfied</font>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Derivation" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_702543024" CREATED="1452672927022" MODIFIED="1459767826004">
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_934334482" CREATED="1452672655899" MODIFIED="1459767794928">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="k" STYLE_REF="Beschreibung" ID="ID_1907863209" CREATED="1452672667899" MODIFIED="1459767798980" HGAP_QUANTITY="46.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1131034498" STARTINCLINATION="61;0;" ENDINCLINATION="61;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1210729663" STARTINCLINATION="72;0;" ENDINCLINATION="72;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="j" STYLE_REF="Beschreibung" ID="ID_143450651" CREATED="1452672683104" MODIFIED="1459767803694" HGAP_QUANTITY="43.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="16" FONT_FAMILY="SansSerif" DESTINATION="ID_577297899" MIDDLE_LABEL="w" STARTINCLINATION="64;0;" ENDINCLINATION="64;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="i" STYLE_REF="Beschreibung" ID="ID_1245790423" CREATED="1452672691114" MODIFIED="1452672847166" HGAP_QUANTITY="58.0 px" VSHIFT_QUANTITY="8.0 px">
<hook NAME="FreeNode"/>
</node>
</node>
</node>
<node TEXT="k" STYLE_REF="Beschreibung" ID="ID_599523705" CREATED="1452672674603" MODIFIED="1459767801043" HGAP_QUANTITY="43.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_143450651" STARTINCLINATION="71;0;" ENDINCLINATION="71;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="j" STYLE_REF="Beschreibung" ID="ID_1131034498" CREATED="1452672677378" MODIFIED="1459767806096" HGAP_QUANTITY="43.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1245790423" STARTINCLINATION="63;0;" ENDINCLINATION="63;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="i" STYLE_REF="Beschreibung" ID="ID_577297899" CREATED="1452672688367" MODIFIED="1452672848102" HGAP_QUANTITY="60.0 px" VSHIFT_QUANTITY="7.0 px">
<hook NAME="FreeNode"/>
</node>
</node>
<node TEXT="j" STYLE_REF="Beschreibung" ID="ID_1210729663" CREATED="1452672680294" MODIFIED="1452672844496" HGAP_QUANTITY="43.0 px">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1245790423" STARTINCLINATION="72;0;" ENDINCLINATION="72;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_577297899" STARTINCLINATION="62;0;" ENDINCLINATION="62;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
</node>
</node>
</node>
<node TEXT="Output k" STYLE_REF="Beschreibung" ID="ID_1523982106" CREATED="1452672938477" MODIFIED="1452673126729">
<hook EQUATION="N_k(x,v,w)=\sum_j v_{jk} \phi (\sum_i w_{ji} x_i)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Loss" STYLE_REF="Beschreibung" ID="ID_12965156" CREATED="1452673140393" MODIFIED="1459767814523">
<hook EQUATION="E = \| z - N(x,v,w) \| = \frac 1 2 \sum_k (\underbrace{z_k - N_k(x,v,w)}_{-\delta_k})^2\\" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Derivative" STYLE_REF="Beschreibung" ID="ID_84722982" CREATED="1452679920411" MODIFIED="1459770661352">
<hook EQUATION="\begin{align}&#xa;&amp;\frac{\partial E}{\partial (v,w)}: &amp;&amp;(1)\;\; \frac{\partial E}{\partial v_{jk}}&amp;&amp;=\delta_k \frac{\partial N_k(x,v,w)}{\partial v_{jk}}\\&#xa;&amp; &amp;&amp; &amp;&amp;=\delta_k \phi(\sum_i w_{ij}x_i)\\&#xa;&amp; &amp;&amp;(2)\;\; \frac{\partial E}{\partial w_{ij}} &amp;&amp;= \sum_k \delta_k \frac{\partial N_k(x,v,w)}{\partial w_{ij}}\\&#xa;&amp; &amp;&amp; &amp;&amp;= \underbrace{\sum_k \delta_k\; v_{jk}}_{k}\; \phi^&apos;(\sum_i w_{ij} x_i)\;x_i&#xa;\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Gradient descent" STYLE_REF="Beschreibung" ID="ID_1554435094" CREATED="1452673225204" MODIFIED="1452673255036">
<hook EQUATION="(v,w)_{t+1}=(v,w)_t - \gamma_k \nabla E_t" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
</node>
</node>
<node TEXT="Optimisation" STYLE_REF="Stichpunkt" FOLDED="true" ID="ID_461971860" CREATED="1452612012832" MODIFIED="1459771240871"><richcontent TYPE="DETAILS" HIDDEN="true">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;We have a model p_w( z | x ). This can, eg.be a neural network. </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">Minimise loss to find better values of w:</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\mathcal L(w)=-\log \prod_i p_w (z_i|x_i)" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="Convex Problem" STYLE_REF="Beschreibung" ID="ID_1378297834" CREATED="1452612188427" MODIFIED="1452612224408"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;good methods exist (e.g. SVD)</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Non-convex Problem" STYLE_REF="Beschreibung" ID="ID_1631920389" CREATED="1452612229522" MODIFIED="1459770789528"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Only incremental methods are known.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Steepest descent&#xa;or gradient descent" STYLE_REF="Stichpunkt" ID="ID_1901341595" CREATED="1452613353801" MODIFIED="1459770793791">
<hook EQUATION="g \equiv \nabla \mathcal L\\&#xa;u_i = -g_i\\&#xa;w_{i+1} = w_i + \alpha u_i" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;greedy algorithm</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Search direction" STYLE_REF="Beschreibung" ID="ID_562845539" CREATED="1452613494709" MODIFIED="1452613504644">
<hook EQUATION="u" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Learning parameter&#xa;or step size" STYLE_REF="Beschreibung" ID="ID_1805405647" CREATED="1452613477657" MODIFIED="1459770797223">
<hook EQUATION="\alpha" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="too small" STYLE_REF="Beschreibung" ID="ID_290976580" CREATED="1452613554485" MODIFIED="1452613586027"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;we find minimum more slowly </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">we end up in local minima or saddle/flat points</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="too large" STYLE_REF="Beschreibung" ID="ID_1909920467" CREATED="1452613589566" MODIFIED="1452613609369"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;you may never find a minimum; oscillations usually occur</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Momentum" STYLE_REF="Beschreibung" ID="ID_1671712691" CREATED="1452614005161" MODIFIED="1459770801842">
<hook EQUATION="\begin{align}&#xa;u_0 &amp;= -g_0\\&#xa;u_i &amp;= -g_i + \beta u_{i-1}\\&#xa;&amp;= -g_i - \beta g_{i-1} - \beta^2 g_{i-2} - \beta^3 g_{i-3} - \ldots\\&#xa;w_{i+1} &amp;= w_i + \alpha u_i&#xa;\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Typically we choose </font></i><font size="1">&#945; </font><i http-equiv="content-type" content="text/html; charset=utf-8"><font color="#666666" size="1">&#160;small and&#160;</font></i><font size="1">&#946; </font><i http-equiv="content-type" content="text/html; charset=utf-8"><font color="#666666" size="1">&#160;large (of course,</font></i><font size="1">&#945; </font><i http-equiv="content-type" content="text/html; charset=utf-8"><font color="#666666" size="1">&#160;,</font></i><font size="1">&#946; </font><i http-equiv="content-type" content="text/html; charset=utf-8"><font color="#666666" size="1">&#160;&gt; 0).</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Learning Rate" STYLE_REF="Beschreibung" ID="ID_1974651844" CREATED="1452614173931" MODIFIED="1452614183276">
<hook EQUATION="\alpha" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Momentum" STYLE_REF="Beschreibung" ID="ID_808638828" CREATED="1452614183786" MODIFIED="1452614195022">
<hook EQUATION="\beta" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
</node>
<node TEXT="Condition of the Hessian" STYLE_REF="Stichpunkt" ID="ID_446496641" CREATED="1452614627440" MODIFIED="1452615011151"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font color="#666666" size="1">&#160;Following the gradient is not always the best choice. Close to minima, it appears that Loss functions are close to quadratic. </font></i>
    </p>
    <p>
      <i><font color="#666666" size="1">A large condition number means that some directions of H are very steep compared to others. In neural networks, a condition of 10^10 is not uncommon. A class of optimisers (CG, Adam, rprop, adadelta, ...) deal with such H.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\mathcal L(x)= \mathcal L(0) + x \underbrace{\frac{\partial \mathcal L}{\partial w}}_g + x^2 \underbrace{\frac{\partial^2 \mathcal L}{\partial w^2}}_{\text{Hessian H}}\\&#xa;\text{Condition} = \text{largest EV} / \text{smallest EV}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
<node TEXT="Practical considerations" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_104477067" CREATED="1453230091413" MODIFIED="1453230099475">
<node TEXT="Deep Learning" STYLE_REF="Stichpunkt" FOLDED="true" ID="ID_1933049014" CREATED="1453231906712" MODIFIED="1453231910192">
<node TEXT="" ID="ID_1294895382" CREATED="1453230285608" MODIFIED="1459771451142">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="universal approximation" STYLE_REF="Beschreibung" ID="ID_779471246" CREATED="1453230114715" MODIFIED="1459771442610"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;an MLP with at least two layers of weights can approximate arbitrarily well any given mapping from one finite input space with a finite number of discontinuities to another, if we have enough hidden units.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Restricted representation and generalisation was one of the biggest problems." STYLE_REF="Beschreibung" ID="ID_819747965" CREATED="1453230323003" MODIFIED="1453230381727"/>
</node>
<node TEXT="parameter finding" STYLE_REF="Beschreibung" ID="ID_24054106" CREATED="1453230197264" MODIFIED="1453230388580">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_819747965" STARTINCLINATION="328;0;" ENDINCLINATION="328;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;finding the optimum set of weights w for an MLP is an NP-complete problem.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="problems" STYLE_REF="Beschreibung" ID="ID_685581226" CREATED="1453230500220" MODIFIED="1459771456456">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="big networks are computationally expensive" STYLE_REF="Beschreibung" ID="ID_1473971305" CREATED="1453230509541" MODIFIED="1453230525267"/>
<node TEXT="a big problem: the vanishing gradient" STYLE_REF="Beschreibung" ID="ID_1567152032" CREATED="1453230402757" MODIFIED="1459771459854"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;the lower you get in the network, the more the gradient vanishes.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\frac{\partial E(w)}{\partial w_{ij}^H}\gg\frac{\partial E(w)}{\partial w_{ij}^{H-1}}\\&#xa;\frac{\partial E(w)}{\partial w_{H-1,i,j}}=\delta_{H-1,j}x_i=\sum_l\underbrace{\delta_{H,l}\times w_{Hlk}x_i}_{small\times small = smaller" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Back-propagation is difficult for multiple layers due to the vanishing gradient" STYLE_REF="Beschreibung" ID="ID_1055366712" CREATED="1453231130870" MODIFIED="1453231151213"/>
<node TEXT="Unsupervised pretraining with Restricted Boltzmann Machines solved that." STYLE_REF="Beschreibung" ID="ID_417459193" CREATED="1453231169516" MODIFIED="1453231190359"/>
</node>
</node>
<node TEXT="However, it was later found that DNN can also be successfully trained:" STYLE_REF="Beschreibung" ID="ID_659320195" CREATED="1453231260415" MODIFIED="1459771463924">
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="use many labelled data (e.g. now well possible for images)" STYLE_REF="Beschreibung" ID="ID_970154918" CREATED="1453231307997" MODIFIED="1453231366426"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;&quot;adding more data&quot;</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="train &quot;longer&quot; (possible with GPUs)" STYLE_REF="Beschreibung" ID="ID_599073853" CREATED="1453231368129" MODIFIED="1453231379333"/>
<node TEXT="better weight initialisation (new methods were developed)" STYLE_REF="Beschreibung" ID="ID_926925178" CREATED="1453231380321" MODIFIED="1453231399965"/>
<node TEXT="regularise with &quot;dropout&quot;" STYLE_REF="Beschreibung" ID="ID_1170402450" CREATED="1453231400681" MODIFIED="1453231409079"/>
</node>
<node TEXT="why do multiple hidden layers improve generalisation?" STYLE_REF="Beschreibung" ID="ID_1593396811" CREATED="1453231469720" MODIFIED="1453231576401"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Functions compactly represented with k layers may require exponential size with k-1 layers. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">Multiple levels of latent variables allow combinatorial sharing of statistical strength. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">Different high-level features share lower-level features.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Rectifier linear units" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1593472281" CREATED="1459771876018" MODIFIED="1459771884670">
<node TEXT="Rectifier linear units" STYLE_REF="Stichpunkt" ID="ID_1293131951" CREATED="1453231956495" MODIFIED="1459771869172"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Here is the trick to make logistic neurons more powerful, but keeping the number of parameters constant: </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">1. make many copies of each neuron </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">2. all neurons have the same parameters, but have a fixed offset to the bias: -1,-0.5, 0.5, ... </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">Apart from saving parameters, this also reduces the vanishing gradient (but there is no guarantee that this improves things)</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\sum_{n=1}^\infty logistic(x+0,5-n)\approx \log(1+e^x)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Dropout" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_592451870" CREATED="1459771890256" MODIFIED="1459771895453">
<node TEXT="Dropout" STYLE_REF="Stichpunkt" ID="ID_577740417" CREATED="1453232265842" MODIFIED="1459771900947"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Lets look at a neural network with one hidden layer. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">Each time a learning sample is learned, we randomly put to 0 each hidden unit with probability 0.5. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">We are therefore randomly sampling from 2^H different architectures, but share the same weights.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="how is the output computed?" STYLE_REF="Beschreibung" ID="ID_446415524" CREATED="1453232475645" MODIFIED="1453232557232"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;To compute the output, one could average all possible models. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">This is too expensive, however. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">Instead we take the half of the outgoing weights to get the same results. This computes the geometric mean of the preditions of all 2^H models.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Quote" STYLE_REF="Beschreibung" ID="ID_902192832" CREATED="1453232559161" MODIFIED="1453232590952"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;If your deep neural network is overfitting, use dropout. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">If it isnt, it will probably be too small.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Batch Learning" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1400405163" CREATED="1459772153414" MODIFIED="1459772164155">
<node TEXT="better algorithm for backprop" STYLE_REF="Stichpunkt" ID="ID_1948423654" CREATED="1453232721674" MODIFIED="1459771308902"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Use batches of data and sum the delta weights for each batch. After use summed delta weights to update weights.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Initialising the weights" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_885171218" CREATED="1453232810579" MODIFIED="1459772346229">
<node TEXT="Randomization" STYLE_REF="Beschreibung" ID="ID_59167425" CREATED="1453232870378" MODIFIED="1453232951527"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;If two hidden units have exactly the same bias and exactly the same incoming and outgoing weights, they will always get exactly the same gradient, so they can never learn to be different features. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">We break symmetry by initialising the weights to have small random values.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Prevent overshooting" STYLE_REF="Beschreibung" ID="ID_71633036" CREATED="1453232979322" MODIFIED="1453233071789"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;If a hidden unit has a big fan-in, small changes on many of its incoming weights can cause the learning to overshoot. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">We generally want smaller incoming weights when the fan-in is big, so initialise the weights to be proportional to sqrt(fan-in). </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">We also can scale the learning rate the same way.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Regularization" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_309362631" CREATED="1453233132287" MODIFIED="1459772356880">
<node TEXT="Shifting and scaling the inputs" STYLE_REF="Stichpunkt" ID="ID_1173596949" CREATED="1453233155284" MODIFIED="1459772174849"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;When using steepest descent, shifting / scaling the input values makes a big difference. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">It usually helps to transform each component of the input vector so that it has zero mean over the whole training set. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">The tanh() produces hidden activations that are roughly zero mean. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">In this respect it is better than the sigmoid.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="A more thorough method: Decorrelate the input components" STYLE_REF="Stichpunkt" ID="ID_24252917" CREATED="1453233377944" MODIFIED="1459772176026"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;For a linear neuron, we get a big win by decorrelating each component of the input from the other input components. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">There are several different ways to decorrelate inputs. A reasonable method is to use Principal Components Analysis (PCA) </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- Drop the principal components with the smallest eigenvalues. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- Divide the remaining principal components by the square roots of their eigenvalues. For a linear neuron, this converts an axis aligned elliptical error surface into a circular one. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">For a circular error surface, the gradient points straight towards the minimum.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="speed-up tricks" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_139291020" CREATED="1453233656732" MODIFIED="1459772651426"><richcontent TYPE="DETAILS" HIDDEN="true">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;If we start with a very large learning rate, weights get very large and one suffers from &quot;saturation&quot; in the neurons. This leads to a vanishing gradient, and learning is stuck.</font></i>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="seperate adaptive learning rates" STYLE_REF="Stichpunkt" ID="ID_1453112769" CREATED="1453233863442" MODIFIED="1459772512401"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;In a multilayer net, the appropriate learning rates can vary widely between weights: </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- The magnitudes of the gradients are often very different for different layers, especially if the initial weights are small. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- The fan-in of a unit determines the size of the &quot;overshoot&quot; effects caused by simultaneously changing many of the incoming weights of a unit to correct the same error. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">=&gt; So use a global learning rate (set by hand) multiplied by an appropriate local gain that is determined empirically for each weight.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="how to determine the individual learning rates" STYLE_REF="Beschreibung" ID="ID_1347782825" CREATED="1453234105106" MODIFIED="1453234231378"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;start with a local gain of 1 for every weight. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">Increase the local gain if the gradient for that weight does not change sign. Use small additive increases and multiplicative decreases (for mini-batch) </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- this ensures that big gains decay rapidly when oscillation start. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- If the gradient is totally random the gain will hover around 1 when we increase by delta half the time and decrease by times 1 - delta halt the time.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="rprop" STYLE_REF="Stichpunkt" ID="ID_1150669721" CREATED="1453234388143" MODIFIED="1459772521081"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;The size of the gradient can be very different for different weights and can change during learning. Also remember the condition of the Hessian. This makes it hard to choose a single global learning rate. </font></i>
    </p>
    <p>
      
    </p>
    <p>
      <i><font size="1" color="#666666">Idea: use only the sign of the gradient. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">The weight updates are all of the same magnitude. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">This escapes from plateaus with tiny gradients quickly. </font></i>
    </p>
    <p>
      
    </p>
    <p>
      <i><font size="1" color="#666666">rprop: This combines the idea of using the sign of the gradient with the idea of adapting the step size separately for each weight. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- Increase the step size for each weight multiplicatively (e.g. times 1.2) if the signs of its last two gradients agree. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- Otherwise decrease the step size multiplicatively (e.g. times 0.5). </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- Dont make the step size too large or too small.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="rmsprop: a mini-batch version of rprop" STYLE_REF="Stichpunkt" ID="ID_1937976168" CREATED="1453234662007" MODIFIED="1459772522314"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;rprop is quivalent to using the gradient but also dividing by the size of the gradient. </font></i>
    </p>
    <p>
      
    </p>
    <p>
      <i><font size="1" color="#666666">The problem with mini-batch rprop is that we divide by a different number for each mini-batch. So why not force the number we divide by to be very similar for adjacent mini-batches? </font></i>
    </p>
    <p>
      
    </p>
    <p>
      <i><font size="1" color="#666666">rmsprop: Keep a moving average of the squared gradient for each weight. Diving the gradient by sqrt ( meanSquare ) makes the learning work much better.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="meanSquare(w,t) = 0.9 \;meanSquare(w,t-1) + 0.1 \left( \frac{\partial E}{\partial w} (t) \right)^2" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="early stopping" STYLE_REF="Beschreibung" ID="ID_1103340611" CREATED="1453234990250" MODIFIED="1459772683469"><richcontent TYPE="DETAILS" HIDDEN="true">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Eary stopping is the standard regularisation technique for MLPs. Typically, use about 10% to 30% of the data for cross-validation.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="hyperparameter optimisation" STYLE_REF="Stichpunkt" FOLDED="true" ID="ID_1323444351" CREATED="1453235276272" MODIFIED="1459772690057"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;We often start finding these by &quot;playing around&quot; with some reasonale estimates. In the end, one often uses hyperparameter optimisation to find a good set. Random search or Bayesian Optimisation are both viable candidates.</font></i>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="number of hidden layers" STYLE_REF="Beschreibung" ID="ID_1271878612" CREATED="1453235371760" MODIFIED="1453235413693"/>
<node TEXT="number of hidden units" STYLE_REF="Beschreibung" ID="ID_138443211" CREATED="1453235377070" MODIFIED="1453235413382"/>
<node TEXT="type of activation function" STYLE_REF="Beschreibung" ID="ID_750658225" CREATED="1453235382595" MODIFIED="1453235413065"/>
<node TEXT="learning method" STYLE_REF="Beschreibung" ID="ID_47974907" CREATED="1453235390788" MODIFIED="1453235412769"/>
<node TEXT="learning parameters" STYLE_REF="Beschreibung" ID="ID_446946408" CREATED="1453235394265" MODIFIED="1453235412472"/>
<node TEXT="data set split" STYLE_REF="Beschreibung" ID="ID_1714868080" CREATED="1453235398364" MODIFIED="1453235412183"/>
<node TEXT="data reprocessing" STYLE_REF="Beschreibung" ID="ID_1402699232" CREATED="1453235404441" MODIFIED="1453235411792"/>
</node>
<node TEXT="FFNN for classification" STYLE_REF="Stichpunkt" ID="ID_786444297" CREATED="1453235458264" MODIFIED="1459772708535"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Use Bernoulli rather than Gaussian assumption. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">When classifying data, use one output per class: one-hot encoding. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">Imagine the other case. If one encodes class 1 as z=0 and class 2 by z=1, then the continuous output is difficult to interpret. What does z=0.5 mean? </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">This gets even worth with 3,... classes. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">We prefer to use a one-hot encoding! Following the previous slide, each output encodes the likelihood of x belonging to that class.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Advanced models" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_368409825" CREATED="1453235957931" MODIFIED="1453235965366">
<node TEXT="convolutional neural networks (cNN)" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1943734193" CREATED="1453236271946" MODIFIED="1453236287470">
<node TEXT="convolutional layer" STYLE_REF="Beschreibung" ID="ID_1944879552" CREATED="1453236341314" MODIFIED="1453236415206"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;a convolutional layer consists of a number of n &#215; m filters which are convolved on the image. For instance, a 5 &#215; 5 filter is matched on all positions of the image, and the result of this match is the next image. This is done for many different filters (see figure);</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="maxpooling layer" STYLE_REF="Beschreibung" ID="ID_1477304340" CREATED="1453236419851" MODIFIED="1453236433643"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;a maxpooling layer downsamples the resulting images, typically in 2 &#215; 2 windows.</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="This structure is stacked, typically 2 to 4 times." STYLE_REF="Beschreibung" ID="ID_1185156015" CREATED="1453236442157" MODIFIED="1453236444858"/>
<node TEXT="The power of a cNN is that the convolution filters are learned." STYLE_REF="Beschreibung" ID="ID_549234799" CREATED="1453236449203" MODIFIED="1453236451633"/>
</node>
<node TEXT="unsupervised learning with neural networks" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1976777852" CREATED="1453236539380" MODIFIED="1453236557078">
<node TEXT="auto-encoder networks" STYLE_REF="Beschreibung" ID="ID_591772037" CREATED="1453236643474" MODIFIED="1453237021339"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Idea: find compact representation of inputs (unsupervised) by </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">1. stacking a second neural network on top of the first </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">2. letting the whole NN compute y(x)=x. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- middle layer (&quot;latents&quot;) usually has fewer neurons. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- latent representation z = compact code for x </font></i>
    </p>
    <p>
      
    </p>
    <p>
      <i><font size="1" color="#666666">These networks make a compact representation of data (&quot;dimensionality reduction&quot;). However, you cannot control the representation!</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="recurrent neural networks" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_832782774" CREATED="1453236860151" MODIFIED="1453236945657"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;adding weights between the neurons within each hidden layer. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">Can be used to represent time series. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">Can be trained with backpropagation-through-time.</font></i>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="applications" STYLE_REF="Beschreibung" ID="ID_1429173719" CREATED="1453236969947" MODIFIED="1453237013823"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;robotics </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">handwritten character recognition and generation </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">text recognition, generation, annotation, etc.; speech recognition </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">machine translation</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="probabilistic neural networks" STYLE_REF="Beschreibung" ID="ID_1477799382" CREATED="1453237048125" MODIFIED="1453237178171"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Each neuron represents a random variable rather than a value. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">The probabilistic neural network introduces many possibilitiies: </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- prediction of confidence intervals at the output; </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- probabilistic version of dropout </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- towards a full probabilistic interpretation; </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">- ...</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Expectation Maximisation" STYLE_REF="Stichpunkt" FOLDED="true" POSITION="right" ID="ID_570358218" CREATED="1453902439482" MODIFIED="1453902448763">
<node TEXT="Face or Not?" STYLE_REF="Beschreibung" ID="ID_1726872413" CREATED="1453903200534" MODIFIED="1459779860215"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;We want to detect faces in an image by sliding a 60 x 60 (pixel) window over the image (at different scales) and classifying every path (face/noface). For training, we have a collection of face patches and non-face patches. A patch x is represented as a 10800 dimenional vector (60 x 60 x 3).</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Generative classifier" STYLE_REF="Beschreibung" ID="ID_1602716778" CREATED="1453903356409" MODIFIED="1453903452184">
<hook EQUATION="p(f|x)=\frac{p(x|f)p(f)}{\sum_f p(x|f)p(f)}\\&#xa;p(x|f=0) = \mathcal N(\mu_0,\Sigma_0)\\&#xa;p(x|f=1) = \mathcal N (\mu_1,\Sigma_1)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
<node TEXT="Mixture of Gaussians" STYLE_REF="Stichpunkt" FOLDED="true" ID="ID_1618669101" CREATED="1453903604425" MODIFIED="1459779915475"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;z is a hidden or latent variable that indicates, from which component k' the sample x was generated (i.e. z can be represented by 1-of-k coding).</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="p(x) = \sum_{m=1}^k p(z=m|\pi)\mathcal N(x|\mu_m,\Sigma_m)\\&#xa;\pi_k \equiv p(z=k|\pi)" NAME="plugins/latex/LatexNodeHook.properties"/>
<node TEXT="Likelihood" STYLE_REF="Stichpunkt" ID="ID_236220896" CREATED="1453903894473" MODIFIED="1459780010951"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;We assume a mixture of k Gaussians and have a set of n i.i.d. observations, D={x1, ..., xn}. </i></font>
    </p>
    <p>
      <font size="1" color="#666666"><i>Note that the algorithm fails to simplify the component density terms - a mixture distribution doest not lie in the exponential family and thus direct optimisation is not easy.</i></font>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="p(\mathcal D|\{ \mu_m \},\{\Sigma_m\},\pi) = \prod_{i=1}^n\sum_{m=1}^k \pi_m \mathcal N(x_i|\mu_m,\Sigma_m)\\&#xa;\Rightarrow l(\theta) = \log p(\mathcal D|\{ \mu_m \},\{\Sigma_m\},\pi) = \sum_{i=1}^n\log\sum_{m=1}^k \pi_m \mathcal N(x_i|\mu_m,\Sigma_m)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Derivatives" STYLE_REF="Beschreibung" ID="ID_1566985708" CREATED="1453904329651" MODIFIED="1459779873594">
<hook EQUATION="\frac{\partial l(\theta)}{\partial \mu_m}=\sum_i r_{im}\Sigma_m^{-1}(x_i-\mu_m)\\&#xa;\frac{\partial l(\theta)}{\partial \Sigma_m}=\sum_i r_{im}(\Sigma_{m}-(x_i-\mu_m)(x_i-\mu_m)^T)\\&#xa;\frac{\partial l(\theta)}{\partial \pi_m} = \sum_i \frac{r_{im}}{\pi_m}\\&#xa;r_{im} = p(z_i=m|x_i) = \frac{\pi_m \mathcal N(x_i|\mu_m,\Sigma_m)}{\sum_k \pi_k \mathcal N(x_i|\mu_k,\Sigma_k)}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Expectation Maximization" STYLE_REF="Stichpunkt" ID="ID_1209336820" CREATED="1453904812904" MODIFIED="1453906041273"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;If we knew what component each point belonged to, we could solve for each component seperately. With this additional information, we would have complete knowledge of the (generative) process, We would optimize</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\log p(\{x_1,\ldots,x_n,z_1,\ldots,z_n\}|\{\mu_m\},\{\Sigma_m\},\pi)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="" STYLE_REF="Beschreibung" ID="ID_1688757478" CREATED="1453906138963" MODIFIED="1453906265261"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Lets just do that! But instead of the assignments for the components we use r_im and solve</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\frac{\partial l(\theta)}{\partial \mu_m}=0,\;\frac{\partial l(\theta)}{\partial \Sigma_m}=0,\;\frac{\partial l(\theta)}{\partial \pi_m}+\lambda=0" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="And then update the responsibilities." STYLE_REF="Beschreibung" ID="ID_495446520" CREATED="1453906270216" MODIFIED="1453906293377"/>
<node TEXT="And then repeat" STYLE_REF="Beschreibung" ID="ID_1977304663" CREATED="1453906286312" MODIFIED="1453906292838"/>
</node>
</node>
<node TEXT="K-Means" STYLE_REF="Stichpunkt" ID="ID_922019806" CREATED="1453909866550" MODIFIED="1459779878606"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;In the E-Step (compute responsibilities), use hard assignment.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\Sigma_k=\sigma I,\;\pi_k=\frac 1 k,\;\mu_k\\&#xa;\text{minimize}\left( \frac 1 n \sum_{i=1}^n \|x_i-\mu_{x_i}\|^2 \right)\\&#xa;\text{with }\mu_{x_i} \equiv \text{ closest }\mu_k \text{ to } x_i" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Initialization" STYLE_REF="Beschreibung" ID="ID_1851650017" CREATED="1453910880061" MODIFIED="1453910906249"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Finding optimal clusters (fixed k ) for K-Means is NP-hard. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">If initialization is done right, there is a theoretical guarantee for the quality of the solution. </font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Jensen&apos;s inequality" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_1810318697" CREATED="1453907185778" MODIFIED="1459780701369">
<hook EQUATION="l(\theta) = \log \int q(\mathcal Z)\frac{p(\mathcal X,\mathcal Z)}{q(\mathcal Z)}d\mathcal Z\geq \int q(\mathcal Z)\log\frac{p(\mathcal X,\mathcal Z|\theta)}{q(\mathcal Z)}d\mathcal Z:=\mathcal F(q,\theta)" NAME="plugins/latex/LatexNodeHook.properties"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;Suppose that direct optimization of log p(X |&#952;) := (&#952;) is difficult, optimizing the complete-data log likelihood log p(X , Z|&#952;) on the other hand is significantly easier. </i></font>
    </p>
    <p>
      <font size="1" color="#666666"><i>After introducing an arbitrary distribution q(Z) defined over the latent variables, one obtains a lower bound on the log likelihood (&#952;) (using Jensen&#8217;s inequality!)</i></font>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="Arbitrary distribution defined over latent variables" STYLE_REF="Beschreibung" ID="ID_794240540" CREATED="1459780546069" MODIFIED="1459780564154">
<hook EQUATION="q(\mathcal Z)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
<node TEXT="Free energy" STYLE_REF="Beschreibung" ID="ID_1140300387" CREATED="1453907859091" MODIFIED="1459780721575">
<hook EQUATION="\begin{align}&#xa;\mathcal F(q,\Theta)&amp;=\int q(\mathcal Z)\log\frac{p(\mathcal X,\mathcal Z|\Theta)}{q(\mathcal Z)}d\mathcal Z\\&#xa;&amp;= \int q(\mathcal Z)\log\frac{p(\mathcal Z|\mathcal X,\Theta)p(\mathcal X|\Theta)}{q(\mathcal Z)}d\mathcal Z\\&#xa;&amp;= \int q(\mathcal Z)\log p(\mathcal X|\Theta)d\mathcal Z + \int q(\mathcal Z)\log\frac{p(\mathcal Z|\mathcal X,\Theta)}{q(\mathcal Z)}d\mathcal Z\\&#xa;&amp;= \log p(\mathcal X|\Theta)\int q(\mathcal Z)d\mathcal Z - KL[q(\mathcal Z)\|p(\mathcal Z|\mathcal X,\Theta)]\\&#xa;&amp;= l(\Theta) - KL[q(\mathcal Z)\|p(\mathcal Z|\mathcal X,\Theta)]&#xa;\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <font size="1" color="#666666"><i>&#160;So instead of dealing with (&#952;), we are trying to maximize F instead (i.e. a lower bound optimization).</i></font>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="E-Step" STYLE_REF="Stichpunkt" ID="ID_228933963" CREATED="1453908429611" MODIFIED="1459779890287"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Maximize F(q, &#952;) with respect to the distribution over the hidden variables, holding the parameters &#952; fixed. From the previous slide we can see immediatley what this acutally means:</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="q^k(\mathcal Z) := \text{argmax}_{q(\mathcal Z)}\; \mathcal F(q(\mathcal Z),\theta^{k-1})\\&#xa;q^k(\mathcal Z) = p(\mathcal Z|\mathcal X,\theta)" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="Optimizing?" STYLE_REF="Beschreibung" ID="ID_1042127166" CREATED="1453909069417" MODIFIED="1453919489148" HGAP_QUANTITY="53.0 px"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;We have to show that the E-Step and M-Step together never decrease </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">the likelihood:</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="l(\Theta^{k-1})=\mathcal F(q^k,\Theta^{k-1})\leq \mathcal F(q^k,\Theta^k)\leq l(\Theta^k)" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
<node TEXT="Free energy" STYLE_REF="Beschreibung" ID="ID_641171726" CREATED="1453907432100" MODIFIED="1459779905542">
<hook EQUATION="\begin{align}&#xa;\mathcal F(q,\Theta) &amp;= \int q(\mathcal Z)\log\frac{p(\mathcal X,\mathcal Z|\Theta)}{q(\mathcal Z)}d\mathcal Z\\&#xa;&amp;= \int q(\mathcal Z)\log p(\mathcal X,\mathcal Z|\Theta)d\mathcal Z - \int q(\mathcal Z)\log q(\mathcal Z)d\mathcal Z\\&#xa;&amp;= \langle\log p(\mathcal X, \mathcal Z|\Theta)\rangle_{q(\mathcal Z)} + H[q]&#xa;\end{align}" NAME="plugins/latex/LatexNodeHook.properties"/>
<hook NAME="AlwaysUnfoldedNode"/>
<node TEXT="M-Step" STYLE_REF="Stichpunkt" ID="ID_904091783" CREATED="1453908739837" MODIFIED="1453919495147">
<arrowlink SHAPE="EDGE_LIKE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1042127166" STARTINCLINATION="366;0;" ENDINCLINATION="366;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Maximize F(q, &#952;) with respect to the parameters &#952;, holding the hidden distribution q(Z) fixed. (The entropy term does not depend on &#952;, so EM is basically a coordinate ascent in F.</font></i>
    </p>
  </body>
</html>
</richcontent>
<hook EQUATION="\Theta^k := \text{argmax}_\Theta \;\mathcal F(q^k(\mathcal Z),\Theta)=\text{argmax}_\Theta \langle\log p(\mathcal X,\mathcal Z|\Theta)\rangle_{q(\mathcal Z)}" NAME="plugins/latex/LatexNodeHook.properties"/>
</node>
</node>
</node>
<node TEXT="Partial EM" STYLE_REF="Beschreibung" ID="ID_1172497855" CREATED="1453909525382" MODIFIED="1453909623688"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;partial E-step: there is no need to use the optimal distribution for the hidden variables. </font></i>
    </p>
    <p>
      <i><font size="1" color="#666666">partial M-step: here also, maximization is not necessary (follow the gradient for some time is enough) </font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Issues" STYLE_REF="Beschreibung" FOLDED="true" ID="ID_426204547" CREATED="1453909698176" MODIFIED="1453909702732">
<node TEXT="Multiple restarts" STYLE_REF="Beschreibung" ID="ID_761855869" CREATED="1453909704290" MODIFIED="1453909718429"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Avoid bad local minima</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Initialization" STYLE_REF="Beschreibung" ID="ID_1210920892" CREATED="1453909719846" MODIFIED="1453909749807"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Random, Random datapoints k-means</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Convergence" STYLE_REF="Beschreibung" ID="ID_1313094190" CREATED="1453909750251" MODIFIED="1453909777549"><richcontent TYPE="DETAILS">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <i><font size="1" color="#666666">&#160;Parameters stop changing, or observed data log likelihood stops increasing</font></i>
    </p>
  </body>
</html>
</richcontent>
</node>
<node TEXT="Singularities" STYLE_REF="Beschreibung" ID="ID_37146579" CREATED="1453909778329" MODIFIED="1453909782846"/>
</node>
</node>
</node>
</map>

* Encoding: UTF-8.

*** replace with path to repository on your machine

CD 'C:\Users\...[ADD PATH TO REPOSITORY]'.


*** load subject-level CSV data

PRESERVE.
SET DECIMAL DOT.

GET DATA  /TYPE=TXT
  /FILE="SAT_subjectLevel.csv"
  /ENCODING='UTF8'
  /DELIMITERS=","
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /DATATYPEMIN PERCENTAGE=95.0
  /VARIABLES=
  V1 2X
  ID A8
  age F3
  gender F1
  group F1
  IDP_PER F10.2
  IDP_ACC F10.2
  IDP_RT F10.2
  IDP_RT_SD F10.2
  SAW_PER F10.2
  SAW_ACC F10.2
  SAW_RT F10.2
  SAW_RT_SD F10.2
  SWM_PER F10.2
  SWM_ACC F10.2
  SWM_RT F10.2
  SWM_RT_SD F10.2
  SAT_RT F10.2
  MeanPD F10.3
  SAT_Total_points F5.0
  SAT_PER F10.2
  subject 4X
  order F3
  model_alpha F10.3
  model_beta F10.3
  model_theta F10.3
  /MAP.
RESTORE.
CACHE.
EXECUTE.
DATASET NAME SAT_subjectLevel WINDOW=FRONT.

ADD VALUE LABELS group 0'YA' 1'OA'.
ADD VALUE LABELS gender 0'male' 1'female'.
ADD VALUE LABELS order 1'normal' 2'reversed'.

VARIABLE LABELS 
    IDP_PER 'IDP_PER (%)' IDP_RT 'IDP_RT (s)' IDP_RT_SD 'IDP_RT_SD (s)' 
    SAW_PER 'SAW_PER (%)' SAW_RT 'SAW_RT (s)' SAW_RT_SD 'SAW_RT_SD (s)' 
    SWM_PER 'SWM_PER (%)' SWM_RT 'SWM_RT (s)' SWM_RT_SD 'SWM_RT_SD (s)' 
    SAT_PER 'SAT_PER (%)' SAT_RT 'SAT_RT (s)' MeanPD 'Mean Planning Depth'.


SAVE OUTFILE='SAT_subjectLevel.sav'
  /COMPRESSED.


*** load condition-level CSV data
    
PRESERVE.
SET DECIMAL DOT.

GET DATA  /TYPE=TXT
  /FILE="SAT_conditionLevel.csv"
  /ENCODING='UTF8'
  /DELIMITERS=","
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /DATATYPEMIN PERCENTAGE=95.0
  /VARIABLES=
  V1 2X
  ID A8
  noise F1
  steps F1
  age F3
  gender F1
  group F1
  IDP_PER F10.2
  IDP_ACC F10.2
  IDP_RT F10.2
  IDP_RT_SD F10.2
  SAW_PER F10.2
  SAW_ACC F10.2
  SAW_RT F10.2
  SAW_RT_SD F10.2
  SWM_PER F10.2
  SWM_ACC F10.2
  SWM_RT F10.2
  SWM_RT_SD F10.2
  SAT_RT F10.2
  MeanPD F10.3
  SAT_Total_points F5.0
  SAT_PER F10.2
  subject 4X
  order F3
  model_alpha F10.3
  model_beta F10.3
  model_theta F10.3
  /MAP.
RESTORE.
CACHE.
EXECUTE.
DATASET NAME SAT_conditionLevel WINDOW=FRONT.

COMPUTE steps=steps - 2. /* dummy-code steps variable.
EXECUTE.

ADD VALUE LABELS noise 0'low noise' 1'high noise'.
ADD VALUE LABELS steps 0'2 steps' 1'3 steps'.
ADD VALUE LABELS group 0'YA' 1'OA'.
ADD VALUE LABELS gender 0'male' 1'female'.
ADD VALUE LABELS order 1'normal' 2'reversed'.

VARIABLE LABELS 
    IDP_PER 'IDP_PER (%)' IDP_RT 'IDP_RT (s)' IDP_RT_SD 'IDP_RT_SD (s)' 
    SAW_PER 'SAW_PER (%)' SAW_RT 'SAW_RT (s)' SAW_RT_SD 'SAW_RT_SD (s)' 
    SWM_PER 'SWM_PER (%)' SWM_RT 'SWM_RT (s)' SWM_RT_SD 'SWM_RT_SD (s)' 
    SAT_PER 'SAT_PER (%)' SAT_RT 'SAT_RT (s)' MeanPD 'Mean Planning Depth'.

SAVE OUTFILE='SAT_conditionLevel.sav'
  /COMPRESSED.


*** group comparison gender.

DATASET ACTIVATE SAT_subjectLevel.
CROSSTABS
  /TABLES=group BY gender
  /FORMAT=AVALUE TABLES
  /STATISTICS=CHISQ 
  /CELLS=COUNT
  /COUNT ROUND CELL.


*** group comparisons standard t-test.

DATASET ACTIVATE SAT_subjectLevel.
T-TEST groupS=group(0 1)
  /MISSING=ANALYSIS
  /VARIABLES=MeanPD model_alpha model_beta model_theta SAT_PER SAW_PER SWM_PER IDP_PER SAT_RT SAW_RT SWM_RT IDP_RT
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).


*** group comparisons non-parametric Mann-Whitney U.
    
DATASET ACTIVATE SAT_subjectLevel.
NPAR TESTS
  /M-W= MeanPD model_alpha model_beta model_theta SAT_PER SAW_PER SWM_PER SAT_RT SAW_RT SWM_RT IDP_RT
    IDP_PER BY group(0 1)
  /MISSING ANALYSIS.


*** Correlations.

DATASET ACTIVATE SAT_subjectLevel.
CORRELATIONS
  /VARIABLES=group IDP_PER SWM_PER SAW_PER SAT_PER  SAT_RT MeanPD model_alpha model_beta model_theta
  /PRINT=TWOTAIL NOSIG FULL
  /MISSING=PAIRWISE.



*** model quality linear regression.

DATASET ACTIVATE SAT_subjectLevel.
REGRESSION
  /DESCRIPTIVES MEAN STDDEV CORR SIG N
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS CI(95) R ANOVA COLLIN TOL CHANGE ZPP
  /CRITERIA=PIN(.05) POUT(.10)
  /NOORIGIN 
  /DEPENDENT SAT_PER
  /METHOD=ENTER model_alpha model_beta model_theta MeanPD
    /SAVE = PRED(LMQ_PRED) RESID(LMQ_RESID).
FORMATS LMQ_PRED(F10.2) LMQ_RESID(F10.2).

*** PLOT: model quality linear regression model residuals.

DATASET ACTIVATE SAT_subjectLevel.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=LMQ_PRED LMQ_RESID MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FITLINE TOTAL=NO SUBgroup=NO
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: LMQ_PRED=col(source(s), name("LMQ_PRED"))
  DATA: LMQ_RESID=col(source(s), name("LMQ_RESID"))
  GUIDE: axis(dim(1), label("Predicted Values"))
  GUIDE: axis(dim(2), label("Residuals"))
  GUIDE: form.line(position(*,0))
  ELEMENT: point(position(LMQ_PRED*LMQ_RESID))
END GPL.

*** PLOT: MeanPD linear regression model historgram residuals.

DATASET ACTIVATE SAT_subjectLevel.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=LMQ_RESID MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: LMQ_RESID=col(source(s), name("LMQ_RESID"))
  GUIDE: axis(dim(1), label("Residuals"))
  GUIDE: axis(dim(2), label("Frequency"))
  ELEMENT: interval(position(summary.count(bin.rect(LMQ_RESID))), shape.interior(shape.square))
  ELEMENT: line(position(density.normal(LMQ_RESID)))
END GPL.

*** PLOT: MeanPD linear regression model Q-Q plot.

DATASET ACTIVATE SAT_subjectLevel.
PPLOT
  /VARIABLES=LMQ_RESID
  /NOLOG
  /NOSTANDARDIZE
  /TYPE=Q-Q
  /FRACTION=BLOM
  /TIES=MEAN
  /DIST=NORMAL.




*** linear regression MeanPD.

DATASET ACTIVATE SAT_subjectLevel.
REGRESSION
  /DESCRIPTIVES MEAN STDDEV CORR SIG N
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS CI(95) R ANOVA COLLIN TOL CHANGE ZPP
  /CRITERIA=PIN(.05) POUT(.10)
  /NOORIGIN 
  /DEPENDENT MeanPD
  /METHOD=BACKWARD group SAT_RT  IDP_PER SWM_PER
  /SAVE = PRED(LMQ_PRED) RESID(LMQ_RESID).
FORMATS LMQ_PRED(F10.2) LMQ_RESID(F10.2).

*** PLOT: MeanPD linear regression model residuals.

DATASET ACTIVATE SAT_subjectLevel.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=LMQ_PRED LMQ_RESID MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FITLINE TOTAL=NO SUBgroup=NO
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: LMQ_PRED=col(source(s), name("LMQ_PRED"))
  DATA: LMQ_RESID=col(source(s), name("LMQ_RESID"))
  GUIDE: axis(dim(1), label("Predicted Values"))
  GUIDE: axis(dim(2), label("Residuals"))
  GUIDE: form.line(position(*,0))
  ELEMENT: point(position(LMQ_PRED*LMQ_RESID))
END GPL.

*** PLOT: MeanPD linear regression model historgram residuals.

DATASET ACTIVATE SAT_subjectLevel.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=LMQ_RESID MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: LMQ_RESID=col(source(s), name("LMQ_RESID"))
  GUIDE: axis(dim(1), label("Residuals"))
  GUIDE: axis(dim(2), label("Frequency"))
  ELEMENT: interval(position(summary.count(bin.rect(LMQ_RESID))), shape.interior(shape.square))
  ELEMENT: line(position(density.normal(LMQ_RESID)))
END GPL.

*** PLOT: MeanPD linear regression model Q-Q plot.

DATASET ACTIVATE SAT_subjectLevel.
PPLOT
  /VARIABLES=LMQ_RESID
  /NOLOG
  /NOSTANDARDIZE
  /TYPE=Q-Q
  /FRACTION=BLOM
  /TIES=MEAN
  /DIST=NORMAL.





*** lme model of MeanPD incl. group*steps and group*noise interactions.

DATASET ACTIVATE SAT_conditionLevel.
MIXED MeanPD WITH group noise steps
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, 
    ABSOLUTE)
  /FIXED=group noise steps group*noise group*steps | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise steps | SUBJECT(ID) COVTYPE(VC)
  /SAVE = PRED(LME_PRED) RESID(LME_RESID).
FORMATS LME_PRED(F10.2) LME_RESID(F10.2). 


*** PLOT: lme model residuals.

DATASET ACTIVATE SAT_conditionLevel.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=LME_PRED LME_RESID MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FITLINE TOTAL=NO SUBgroup=NO
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: LME_PRED=col(source(s), name("LME_PRED"))
  DATA: LME_RESID=col(source(s), name("LME_RESID"))
  GUIDE: axis(dim(1), label("Predicted Values"))
  GUIDE: axis(dim(2), label("Residuals"))
  GUIDE: form.line(position(*,0))
  ELEMENT: point(position(LME_PRED*LME_RESID))
END GPL.

*** PLOT: lme model historgram residuals.

DATASET ACTIVATE SAT_conditionLevel.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=LME_RESID MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: LME_RESID=col(source(s), name("LME_RESID"))
  GUIDE: axis(dim(1), label("Residuals"))
  GUIDE: axis(dim(2), label("Frequency"))
  ELEMENT: interval(position(summary.count(bin.rect(LME_RESID))), shape.interior(shape.square))
  ELEMENT: line(position(density.normal(LME_RESID)))
END GPL.

*** PLOT: lme model Q-Q plot.

DATASET ACTIVATE SAT_conditionLevel.
PPLOT
  /VARIABLES=LME_RESID
  /NOLOG
  /NOSTANDARDIZE
  /TYPE=Q-Q
  /FRACTION=BLOM
  /TIES=MEAN
  /DIST=NORMAL.
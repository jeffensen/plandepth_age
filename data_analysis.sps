* Encoding: UTF-8.
CD 'P:\10129\Abschlussarbeiten\JSteffen_Masterarbeit\publication\Suppl_Files\data'.

GET  FILE= 'SAT_subjectLevel_and_cognTasks.sav'
    /KEEP  Age Sex Group SAT_PER RT_1st_choice MeanPD_1st_choice Model_alpha Model_beta Model_theta IDP_PER SWM_PER SAW_PER SAW_RT SWM_RT IDP_RT.
DATASET NAME Paper_Subjects WINDOW=FRONT.

GET  FILE= 'SAT_conditionLevel_with_covariates.sav'
    /KEEP  Subject_ID Group noise steps MeanPD.
DATASET NAME Paper_SAT_Conditions WINDOW=FRONT.

DATASET ACTIVATE Paper_Subjects.


*** Correlations.

CORRELATIONS
  /VARIABLES=Group IDP_PER SWM_PER SAW_PER SAT_PER  RT_1st_choice MeanPD_1st_choice Model_alpha Model_beta Model_theta
  /PRINT=TWOTAIL NOSIG FULL
  /MISSING=PAIRWISE.


*** model quality linear regression.

REGRESSION
  /DESCRIPTIVES MEAN STDDEV CORR SIG N
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS CI(95) R ANOVA COLLIN TOL CHANGE ZPP
  /CRITERIA=PIN(.05) POUT(.10)
  /NOORIGIN 
  /DEPENDENT SAT_PER
  /METHOD=BACKWARD Model_alpha Model_beta Model_theta MeanPD_1st_choice.


*** linear regression MeanPD.

REGRESSION
  /DESCRIPTIVES MEAN STDDEV CORR SIG N
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS CI(95) R ANOVA COLLIN TOL CHANGE ZPP
  /CRITERIA=PIN(.05) POUT(.10)
  /NOORIGIN 
  /DEPENDENT MeanPD_1st_choice
  /METHOD=BACKWARD Group RT_1st_choice  IDP_PER SWM_PER.


*** group comparison gender.

CROSSTABS
  /TABLES=Group BY Sex
  /FORMAT=AVALUE TABLES
  /STATISTICS=CHISQ 
  /CELLS=COUNT
  /COUNT ROUND CELL.


*** group comparisons standard t-test.

T-TEST GROUPS=Group(0 1)
  /MISSING=ANALYSIS
  /VARIABLES=MeanPD_1st_choice Model_alpha Model_beta Model_theta SAT_PER SAW_PER SWM_PER IDP_PER RT_1st_choice SAW_RT SWM_RT IDP_RT
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).


*** group comparisons bootstrap.
    
BOOTSTRAP
  /SAMPLING METHOD=SIMPLE
  /VARIABLES TARGET=MeanPD_1st_choice Model_alpha Model_beta Model_theta SAT_PER SAW_PER SWM_PER RT_1st_choice SAW_RT SWM_RT IDP_RT
    IDP_PER INPUT=Group 
  /CRITERIA CILEVEL=95 CITYPE=BCA  NSAMPLES=10000
  /MISSING USERMISSING=EXCLUDE.
T-TEST GROUPS=Group(0 1)
  /MISSING=ANALYSIS
  /VARIABLES=MeanPD_1st_choice Model_alpha Model_beta Model_theta SAT_PER SAW_PER SWM_PER IDP_PER RT_1st_choice SAW_RT SWM_RT IDP_RT
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).


*** lme model1 of MeanPD.

DATASET ACTIVATE Paper_SAT_Conditions.
MIXED MeanPD WITH Group noise steps
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(10) SCORING(1) 
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)    
  /FIXED=Group noise steps | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise steps | SUBJECT(Subject_ID) COVTYPE(VC)
  /SAVE=PRED RESID.

*** lme model2 of MeanPD incl. group*steps and group*noise interactions.
MIXED MeanPD WITH Group noise steps
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, 
    ABSOLUTE)
  /FIXED=Group noise steps Group*noise Group*steps | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT noise steps | SUBJECT(Subject_ID) COVTYPE(VC).


*** PLOT: lme model1 residuals.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=PRED_1 RESID_1 MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FITLINE TOTAL=NO SUBGROUP=NO
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: PRED_1=col(source(s), name("PRED_1"))
  DATA: RESID_1=col(source(s), name("RESID_1"))
  GUIDE: axis(dim(1), label("Predicted Values"))
  GUIDE: axis(dim(2), label("Residuals"))
  ELEMENT: point(position(PRED_1*RESID_1))
END GPL.

*** PLOT: lme model1 historgram residuals.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=RESID_1 MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: RESID_1=col(source(s), name("RESID_1"))
  GUIDE: axis(dim(1), label("Residuals"))
  GUIDE: axis(dim(2), label("Frequency"))
  ELEMENT: interval(position(summary.count(bin.rect(RESID_1))), shape.interior(shape.square))
END GPL.

*** PLOT: lme model1 Q-Q plot.
PPLOT
  /VARIABLES=RESID_1
  /NOLOG
  /NOSTANDARDIZE
  /TYPE=Q-Q
  /FRACTION=BLOM
  /TIES=MEAN
  /DIST=NORMAL.

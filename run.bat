@echo off
setlocal

set BASEDIR=.\lab_resources\DDI

REM Start CoreNLP server in background
echo Starting CoreNLP server...
start /B java -mx5g -cp ".\stanford-corenlp-4.5.10\*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -quiet true -port 9000 -timeout 15000

timeout /t 1 /nobreak > nul

REM Extract features (devel and train sequentially; bash original ran devel in background)
echo Extracting features...
python extract-features.py %BASEDIR%\data\devel\ > devel.cod
if errorlevel 1 goto error

python extract-features.py %BASEDIR%\data\train\ > train.cod
if errorlevel 1 goto error

REM Equivalent of: tee train.cod | cut -f4- > train.cod.cl
REM (train.cod already saved above, now cut from field 4 onwards)
python -c "data=open('train.cod').readlines(); open('train.cod.cl','w').writelines('\t'.join(l.rstrip('\n').split('\t')[3:])+'\n' for l in data)"
if errorlevel 1 goto error

REM Kill CoreNLP server
echo Stopping CoreNLP server...
taskkill /F /IM java.exe > nul 2>&1

REM Train model
echo Training model...
python train-sklearn.py model.joblib vectorizer.joblib < train.cod.cl
if errorlevel 1 goto error

REM Run model
echo Running model...
python predict-sklearn.py model.joblib vectorizer.joblib < devel.cod > devel.out
if errorlevel 1 goto error

REM Evaluate results
echo Evaluating results...
python evaluator.py DDI %BASEDIR%\data\devel\ devel.out > devel.stats
if errorlevel 1 goto error

echo.
echo Done!
goto end

:error
echo.
echo ERROR: something failed. Check the output above.

:end
pause

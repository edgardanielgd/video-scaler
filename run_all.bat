@echo on
for /f %%i in ('pkg-config --cflags --libs opencv4') do  (
    echo %%i
    set RESULT=%RESULT%
)
echo %RESULT%
g++ -fopenmp main.cpp -o main %RESULT%
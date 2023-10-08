
g++ -fopenmp main.cpp -o main %CD%\$(pkg-config --cflags --libs opencv4)
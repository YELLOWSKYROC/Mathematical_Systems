#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

union DoubleToInt {
  double dVal;
  uint64_t iVal;
};

/*
  A function that rounds a binary64 value to a binary32 value
  stochastically. Implemented by treating FP number representations
  as integer values. 
*/

float SR(double x) {
  union DoubleToInt temp;
  temp.dVal = x;
  uint32_t r = rand() & 0x1FFFFFFF;
  temp.iVal += r;
  temp.iVal = temp.iVal & 0xFFFFFFFFE0000000;
  return (float)temp.dVal;
}


/* --------------------------------- */
/*              PART 1               */
/* --------------------------------- */

// Implement SR_alternative according to the Eqn 1.
float SR_alternative(double x) {
  // P is a random number from 0 to 1
  double P = (double)rand() / (double)((unsigned) RAND_MAX+1);

  // Calculate the neighbouring binary32 values.
  float closest = (float)x;
  float RZ, RA;

  // Different Situation Analysis
  if (x > 0) {
    if (closest > x) {
      RZ = nextafterf(closest, -INFINITY);
      RA = closest;
    }
    else {
      RZ = closest;
      RA = nextafterf(closest, INFINITY);
    }
  }
  else {
    if (closest > x) {
      RZ = closest;
      RA = nextafterf(closest, -INFINITY);
    }
    else {
      RZ = nextafterf(closest, INFINITY);
      RA = closest;
    }
  }

  // Calculate the probability p.
  double p = fabs(x - RZ) / fabs(RA - RZ);
  if (P < p) {
    return RA;
  }
  else {
    return RZ;
  }
}

// Fast two sum algorithm
void fastTwoSum (float a, float b, float *s, float *t) {
  float temp;
  *s = a + b;
  temp = *s - a;
  *t = b - temp;
}

// Implement matrix_multiply function for Part3.
void matrix_multiply(int N, double A[N][N], double B[N][N], double C_32[N][N], double C_32_alt[N][N], double C_comp[N][N], double C_64[N][N]) {
  int i, j, k;
  float sum_32;
  float sum_32_alt;
  float sum_comp;
  double sum_64;

  // Matrix multiplication
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      sum_32 = 0;
      sum_32_alt = 0;
      sum_comp = 0;
      sum_64 = 0;
      float t = 0;
      for (k = 0; k < N; k++) {
        // Binary32 RN
        sum_32 += A[i][k] * B[k][j];

        // Binary32 SR
        sum_32_alt = SR_alternative((double)sum_32_alt+(double)(A[i][k] * B[k][j]));

        // Binary32 Compensated sum
        float b = (A[i][k] * B[k][j]) + t;
        fastTwoSum(sum_comp, b, &sum_comp, &t);

        // Binary64
        sum_64 += A[i][k] * B[k][j];
      }
      C_32[i][j] = sum_32;
      C_32_alt[i][j] = sum_32_alt;
      C_comp[i][j] = sum_comp;
      C_64[i][j] = sum_64;
    }
  }
}

/* --------------------------------- */
const long int K = 5000000;

int main() {
  printf("---------------Part1---------------\n");
  // An arbitrary value for rounding.
  double sample = M_PI;
  double avg = 0;
  double avg_alternative = 0;

  // Calculate the neighbouring binary32 values. 
  float closest = (float)sample;
  float down, up;

  // RA and RZ
  if (closest > sample) {
    down = nextafterf(closest, -INFINITY);
    up = closest;
  }
  else {
    down = closest;
    up = nextafterf(closest, INFINITY);
  }

  // Round many times, and calculate the average values as well as count the 
  // numbers of times rounding was up/down.the numbers of times rounding was up/down.
  int countup = 0;
  int countdown = 0;
  int countup_alternative = 0;
  int countdown_alternative = 0;

  // Open a data file to store the data
  FILE* data_file1 = fopen("data_SR.txt", "w");
  FILE* data_file2 = fopen("data_SRA.txt", "w");

  // Round K times, and calculate the average values as well as count the
  // numbers of times rounding was up/down.
  for (int i = 1; i <= K; i++) {
    // SR rounding
    float rounded1 = SR(sample);
    avg += rounded1;
    if (rounded1 > sample) {
      countup++;
    }
    else {
      countdown++;
    }

    // SR_alternative rounding
    float rounded2 = SR_alternative(sample);
    avg_alternative += rounded2;
    if (rounded2 > sample) {
      countup_alternative++;
    }
    else {
      countdown_alternative++;
    }

    // Print the average error every 1000 times to txt file.
    if(i % 1000 == 0){
      fprintf(data_file1, "%d %.60f\n", i, fabs(avg/i-sample));
      fprintf(data_file2, "%d %.60f\n", i, fabs(avg_alternative/i-sample));
    }
  }
  // Calculate the average value.
  avg = avg/K;
  avg_alternative = avg_alternative/K;

  // Close the data file
  fclose(data_file1);
  fclose(data_file2);

  // Open a pipe to gnuplot and send commands to plot the graph
  FILE* gnuplot_pipe_1 = popen("gnuplot -persistent", "w");
  fprintf(gnuplot_pipe_1, "set xlabel 'Iteration'\n");
  fprintf(gnuplot_pipe_1, "set ylabel 'Absolute Error'\n");
  fprintf(gnuplot_pipe_1, "plot 'data_SR.txt' using 1:2 with lines title 'Avg Error of SR'  lc 'red', \
  'data_SRA.txt' using 1:2 with lines title 'Avg Error of SRA' lc 'blue'\n");
  pclose(gnuplot_pipe_1);

  // Print out some useful stats.
  printf("Value being rounded:               %.60f \n", sample);
  printf("SR average value:                  %.60f \n", avg);
  printf("SR_alternative average value:      %.60f \n", avg_alternative);
  printf("Closest binary32:                  %.60f \n", closest);
  printf("Binary32 SR value before:          %.60f \n", down);
  printf("Binary32 SR value after:           %.60f \n", up);


  // Print out the average of all rounded values
  // Check that SR_alternative function is correct by comparing the probabilities of rounding up/down, 
  // and the expected probability. Print them out below.
  float SRA_up_p, SRA_down_p, SR_up_p, SR_down_p;
  SRA_up_p = (float)countup_alternative / K;
  SRA_down_p = (float)countdown_alternative / K;
  SR_up_p = (float)countup / K;
  SR_down_p = (float)countdown / K;

  // Print out some useful stats.
  printf("Round Up Expected probability:     %f%% \n", (fabs(sample - down)/fabs(up - down))*100);
  printf("Round Down Expected probability:   %f%% \n", (fabs(sample - up)/fabs(up - down))*100);
  printf("SR Probability of rounding up:     %f%% \n", SR_up_p*100);
  printf("SR Probability of rounding down:   %f%% \n", SR_down_p*100);
  printf("SRA Probability of rounding up:    %f%% \n", SRA_up_p*100);
  printf("SRA Probability of rounding down:  %f%% \n", SRA_down_p*100);
  printf("\n");


  /* --------------------------------- */
  /*              PART 2               */
  /* --------------------------------- */
  printf("---------------Part2---------------\n");
  long int N = 500000000;
  float fharmonic = 0;
  float fharmonic_sr = 0;
  float fharmonic_comp = 0;
  double dharmonic = 0;

  // Open a data file to store the data
  FILE* data21 = fopen("data_2_32.txt", "w");
  FILE* data22 = fopen("data_2_sr.txt", "w");
  FILE* data23 = fopen("data_2_comp.txt", "w");

  // Stagnation checker
  float stagnation1 = 0;
  float stagnation2 = 0;

  // For fastTwoSum
  float t = 0;

  for (int i = 1; i <= N; i++) {
    // Recursive sum, binary32 RN
    fharmonic += (float)1/i;

    // Stagnation checker
    if(stagnation1-fharmonic == 0 && stagnation2 == 0){ 
      printf("The value is stagnated at step: %d \n", i-1);
      stagnation2 = 1; 
    }
    stagnation1 = fharmonic;

    // Recursive sum, SR_alternative
    fharmonic_sr = SR_alternative((double)1/i + (double)fharmonic_sr);

    // Recursive sum, compensated
    float b = (float)1/i + t;
    fastTwoSum(fharmonic_comp, b, &fharmonic_comp, &t);

    // Recursive sum, double, binary64
    dharmonic += (double)1/i;

    // Print the average error every 1000000 times to txt file.
    if(i % 1000000 == 0){
      // Calculate the error
      double err1 = fabs(fharmonic - dharmonic);
      double err2 = fabs(fharmonic_sr - dharmonic);
      double err3 = fabs(fharmonic_comp - dharmonic);
      // Print the error to the each data file.
      fprintf(data21, "%d %.60f\n", i, err1);
      fprintf(data22, "%d %.60f\n", i, err2);
      fprintf(data23, "%d %.60f\n", i, err3);
    }
  }

  // Close the data file
  fclose(data21);
  fclose(data22);
  fclose(data23);

  //Open a pipe to gnuplot and send commands to plot the graph
  FILE* gnuplot_pipe_2 = popen("gnuplot -persistent", "w");
  fprintf(gnuplot_pipe_2, "set logscale x\n");
  fprintf(gnuplot_pipe_2, "set logscale y\n");
  fprintf(gnuplot_pipe_2, "set xlabel 'Iteration'\n");
  fprintf(gnuplot_pipe_2, "set ylabel 'Absolute Error'\n");
  fprintf(gnuplot_pipe_2, "plot 'data_2_32.txt' using 1:2 with lines title 'Error of Recursive summation' lc 'red', \
  'data_2_sr.txt' using 1:2 with lines title 'Error of Recursive summation with SR' lc 'orange', \
  'data_2_comp.txt' using 1:2 with lines title 'Error of Compensated summation' lc 'blue'\n");
  pclose(gnuplot_pipe_2);

  // Calculate the absolute error between the binary32 and binary64 values.
  float abs_error = fabs(fharmonic - dharmonic);
  float abs_error_sr = fabs(fharmonic_sr - dharmonic);
  float abs_error_comp = fabs(fharmonic_comp - dharmonic);

  // Print out some useful stats.
  printf("Values of the harmonic series after %ld iterations \n", N);
  printf("Recursive summation, binary32:                             %.30f \n", fharmonic);
  printf("Recursive summation with SR, binary32:                     %.30f \n", fharmonic_sr);
  printf("Compensated summation, binary32:                           %.30f \n", fharmonic_comp);
  printf("Recursive summation, binary64:                             %.30f \n", dharmonic);

  // Absolute error between the binary32 and binary64 values.
  printf("Absolute Error of Recursive summation, binary32:           %.30f \n", abs_error);
  printf("Absolute Error of Recursive summation with SR, binary32:   %.30f \n", abs_error_sr);
  printf("Absolute Error of Compensated summation, binary32:         %.30f \n", abs_error_comp);
  printf("\n");


  /* --------------------------------- */
  /*              PART 3               */
  /* --------------------------------- */
  printf("---------------Part3---------------\n");
  int i, j, n;

  double abs_error_64_32;
  double abs_error_64_32_alt;
  double abs_error_64_comp;

  double sum_64_32 = 0;
  double sum_64_32_alt = 0;
  double sum_64_comp = 0;

  double sum_matrix_32 = 0;
  double sum_matrix_32_alt = 0;
  double sum_matrix_comp = 0;
  double sum_matrix_64 = 0;

  // Open a data file to store the data
  FILE* data_32 = fopen("data_3_32.txt", "w");
  FILE* data_SR = fopen("data_3_SR.txt", "w");
  FILE* data_comp = fopen("data_3_comp.txt", "w");

  // Initialize matrices A and B
  for (n = 1; n <= 100; n++) {
    int N = n;
    abs_error_64_32= 0;
    abs_error_64_32_alt = 0;
    abs_error_64_comp = 0;

    double A[N][N], B[N][N], C_32[N][N], C_32_alt[N][N], C_comp[N][N], C_64[N][N];

    // Initialize the matrices
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        if (i == 0 && j == 0) {
          A[i][j] = 1.0;
          B[i][j] = 1.0;
        }else {
          A[i][j] = 1.0/(i+j+30000000.0);
          B[i][j] = 1.0/(i+j+30000000.0);
        }
      }
    }

    // Multiply the matrices
    matrix_multiply(N, A, B, C_32, C_32_alt, C_comp, C_64);

    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        // Sum the elements of the matrices
        sum_matrix_32 += C_32[i][j];
        sum_matrix_32_alt += C_32_alt[i][j];
        sum_matrix_comp += C_comp[i][j];
        sum_matrix_64 += C_64[i][j];

        // Calculate the absolute error
        abs_error_64_32 += fabs(C_32[i][j] - C_64[i][j]);
        abs_error_64_32_alt += fabs(C_32_alt[i][j] - C_64[i][j]);
        abs_error_64_comp += fabs(C_comp[i][j] - C_64[i][j]);
      }
    }

    // Sum the absolute error
    sum_64_32 += abs_error_64_32;
    sum_64_32_alt += abs_error_64_32_alt;
    sum_64_comp += abs_error_64_comp;

    // Write the data to the each files.
    fprintf(data_32, "%d %.60f\n", n, abs_error_64_32);
    fprintf(data_SR, "%d %.60f\n", n, abs_error_64_32_alt);
    fprintf(data_comp, "%d %.60f\n", n, abs_error_64_comp);
  }

  // Close the files
  fclose(data_32);
  fclose(data_SR);
  fclose(data_comp);

  // Open a pipe to gnuplot and send commands to plot the graph
  FILE* gnuplot_pipe_3 = popen("gnuplot -persistent", "w");
  fprintf(gnuplot_pipe_3, "set xlabel 'Size of matrix'\n");
  fprintf(gnuplot_pipe_3, "set ylabel 'Absolute Error'\n");
  fprintf(gnuplot_pipe_3, "plot 'data_3_32.txt' using 1:2 with lines title 'binary 32' lc 'red', \
  'data_3_SR.txt' using 1:2 with lines title 'SR' lc 'blue', \
  'data_3_comp.txt' using 1:2 with lines title 'Comp' lc 'green'\n");
  pclose(gnuplot_pipe_3);

  // Print out some useful stats.
  printf("Recursive summation, binary32:          %.60f \n", sum_matrix_32);
  printf("Recursive summation with SR, binary32:  %.60f \n", sum_matrix_32_alt);
  printf("Compensated summation, binary32:        %.60f \n", sum_matrix_comp);
  printf("Recursive summation, binary64:          %.60f \n", sum_matrix_64);
  printf("sum_error_64_32:                        %.60f \n", sum_64_32);
  printf("sum_error_64_32_alt:                    %.60f \n", sum_64_32_alt);
  printf("sum_error_64_comp:                      %.60f \n", sum_64_comp);

  return 0;
}
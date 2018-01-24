
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "utils.h"
#include "rlglue/RL_glue.h"

#define	N_EPISODES_PER_RUN	100
#define	N_RUNS			100
#define	N_STEPS_PER_EPISODE	0

void saveResults(double* data, int dataSize, const char* filename);


int main(int argc, char *argv[]) {

	srand(time(NULL));

	double n_steps_to_goal[N_EPISODES_PER_RUN];

	for(int i = 0; i < N_EPISODES_PER_RUN; i++)
		n_steps_to_goal[i] = 0.0;

	for(int i = 0; i < N_RUNS; i++){
		RL_init();

		for(int j = 0; j < N_EPISODES_PER_RUN; j++){
			RL_episode(N_STEPS_PER_EPISODE);
			//printf("Episode Over\n");
			n_steps_to_goal[j] += RL_num_steps();
		}

		printf(".");
	}
	printf("\nDone\n");

	for(int i = 0; i < N_EPISODES_PER_RUN; i++){
		n_steps_to_goal[i] /= (double)N_RUNS;
	}
  
  	/* Save data to a file */
	saveResults(n_steps_to_goal, N_EPISODES_PER_RUN, "RL_EXP_OUT.dat");
  
  	return 0;
}


void saveResults(double* data, int dataSize, const char* filename) {
  FILE *dataFile;
  int i;
  dataFile = fopen(filename, "w");
  for(i = 0; i < dataSize; i++){
    fprintf(dataFile, "%lf\n", data[i]);
  }
  fclose(dataFile);
}
